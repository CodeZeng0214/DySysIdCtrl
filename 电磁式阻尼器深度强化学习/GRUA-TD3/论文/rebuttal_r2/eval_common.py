"""Reviewer #3 (round 2) - shared evaluation utilities.

Evaluation protocol reproduced from the code that produced the published
tables:

  metric   : mean of the local peaks of |x2(t)| found in the last 30 % of the
             episode (scipy.find_peaks, distance=10, prominence=1e-4).  This is
             the "mean maximum amplitude" column of Tables 4-7.
  rollout  : env.run_episode_with_controller(), i.e. the controller is engaged
             at the first peak after 30 % of the simulation time.
  delay    : the loop delay is sampled from N(DELAY_MEAN, DELAY_STD) steps by
             the environment itself, exactly as in ElectromagneticDamperEnv.step().
             This is the setting behind every published number; the label of
             the manuscript is left untouched on request of the author.

A controller whose input dimension is 5 was trained with the measured delay in
the observation, one with dimension 4 was not.  Controllers are rebuilt from
the shapes stored in their own checkpoint, so the MLP runs that differ in width
or depth (T1 / T1-1 / T1-2, T3 / T3-1 / T3-2) are each evaluated with the
network they were actually trained with.
"""
from __future__ import annotations

import os
import sys
import numpy as np
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import build_env                            # noqa: E402
from controller import TD3Controller                 # noqa: E402
from fx import zero, sin_wave, mls_force, tolerance_smooth_reward  # noqa: E402

# Inference is a batch-1 forward pass per simulation step, which is faster on
# the CPU than on the GPU.  Set EVAL_DEVICE=cpu to use the CPU.
if os.environ.get("EVAL_DEVICE", "cuda").lower() == "cpu":
    import torch
    import torch as _torch
    import networks as _networks
    import controller as _controller
    _torch.set_num_threads(1)
    _networks.device = _torch.device("cpu")
    _controller.device = _torch.device("cpu")

DELAY_MEAN = 4
DELAY_STD = 1
OBS_INDICES = [0, 2, 3, 5]

CONDITIONS = ("free", "fixed", "mls")

# No Control numbers of the published tables, used only as a sanity check
PUBLISHED_NC = {("free", False): 4.60e-3, ("free", True): 4.60e-3,
                ("fixed", False): 6.50e-3, ("fixed", True): 6.12e-3,
                ("mls", False): 9.50e-3, ("mls", True): 9.21e-3}


def make_env(condition: str, delay: bool = True, obs_delay: bool | None = None,
             sys_params: dict | None = None, dt_noise: bool | None = None,
             delay_mean: int = DELAY_MEAN, delay_std: int = DELAY_STD,
             seed: int | None = None):
    """Build the environment of one experimental condition.

    condition : 'free'  -> initial displacement 0.006 m, T = 1.2 s  (jitter on)
                'fixed' -> 30 Hz, F0 = 150 N,        T = 2.5 s  (jitter off)
                'mls'   -> MLS random force,         T = 2.0 s  (jitter on)
    delay     : True  -> non-ideal environment (loop delay on)
    obs_delay : whether the measured delay is appended to the observation.
                Defaults to `delay`; must match the input dimension of the
                controller under test.
    """
    if condition == "free":
        T, state0, f_func = 1.2, np.array([0.0, 0.0, 0.0, 0.006, 0.0, 0.0]), zero
        if dt_noise is None:
            dt_noise = True
    elif condition == "fixed":
        T, state0, f_func = 2.5, np.zeros(6), sin_wave(150, 30)
        if dt_noise is None:
            dt_noise = False
    elif condition == "mls":
        T, state0, f_func = 2.0, np.zeros(6), mls_force(amplitude=200, frequency=30, phase=0.3)
        if dt_noise is None:
            dt_noise = True
    else:
        raise ValueError(condition)

    if obs_delay is None:
        obs_delay = delay

    params = {
        'Ts': 0.001,
        'T': T,
        'state0': state0,
        'obs_indices': OBS_INDICES,
        'x1_limit': 0.03,
        'use_dt_noise': bool(dt_noise),
        'dt_noise_std': 0.1,
        'delay_enabled': bool(delay),
        'delay_mean_steps': delay_mean if delay else 0,
        'delay_std_steps': delay_std if delay else 0,
        'include_dt_in_obs': False,
        'include_delay_in_obs': bool(obs_delay),
        'z_func': zero,
        'r_func': tolerance_smooth_reward(1e-3),
        'f_func': f_func,
    }
    if seed is not None:
        np.random.seed(seed)
    return build_env(params, sys_params=sys_params), params


def env_setting(condition: str, delay: bool, state_dim: int, **kw):
    """Environment matching the observation space of a given controller."""
    return make_env(condition, delay=delay, obs_delay=(state_dim == 5), **kw)


# ----------------------------------------------------------------------------
def mean_peak_amplitude(x2: np.ndarray, tail: float = 0.3) -> float:
    """Mean of the local peaks of |x2| inside the last `tail` fraction."""
    x2 = np.asarray(x2, dtype=float)
    start = int(len(x2) * (1.0 - tail))
    amp = np.abs(x2[start:])
    idx, _ = find_peaks(amp, distance=10, prominence=1e-4)
    if len(idx) == 0:
        return float(amp.max())
    return float(amp[idx].mean())


def rollout(env, controller=None, mode="start"):
    """One episode; returns the full state history array [N, 6].

    mode='start'   : the controller acts from the first sampling instant
                     (env.run_episode) - this is the protocol of the paper.
    mode='delayed' : the controller is engaged at the first peak after 30 % of
                     the simulation time (env.run_episode_with_controller).
    """
    if mode == "delayed":
        rec = env.run_episode_with_controller(
            controller=controller,
            state0=env.state0,
            z_func=env.z_func,
            f_func=env.f_func,
        )
    else:
        rec = env.run_episode(
            controller=controller,
            state0=env.state0,
            z_func=env.z_func,
            f_func=env.f_func,
        )
    return rec.as_numpy(keys='state_history')


def attenuation(a_ctrl: float, a_nc: float) -> float:
    """Attenuation rate in percent."""
    return 100.0 * (a_nc - a_ctrl) / a_nc if a_nc > 0 else float('nan')


# ----------------------------------------------------------------------------
def build_td3(arch: str, state_dim: int, freeze_attention: bool = False) -> TD3Controller:
    """Instantiate a TD3 controller with the hyper-parameters of the paper."""
    params = dict(
        arch=arch, norm=False, simple_nn=False,
        state_dim=state_dim, action_dim=1, action_bound=5.0,
        gru_hidden=64, gru_layers=1, hidden_dim=128,
        seq_len=36, fc_seq_len=4,
        actor_lr=5e-6, critic_lr=1e-5, clip_grad=1.0, tau=0.001,
        gamma=0.99, policy_noise=0.2, noise_clip=0.5, policy_freq=3,
        freeze_attention=freeze_attention,
    )
    return TD3Controller(**params)


def _sequential_from_shapes(shapes, out_activation=None):
    """Rebuild a plain MLP from the list of Linear weight shapes."""
    import torch.nn as nn
    layers = []
    for i, (out_f, in_f) in enumerate(shapes):
        layers.append(nn.Linear(in_f, out_f))
        if i < len(shapes) - 1:
            layers.append(nn.ReLU())
        elif out_activation is not None:
            layers.append(out_activation)
    return nn.Sequential(*layers)


def _linear_shapes(state_dict):
    """Linear layers of a Sequential, in order, as (out, in) shape pairs."""
    keys = [k for k in state_dict if k.endswith('.weight')]
    keys.sort(key=lambda k: int(k.split('.')[1]))
    return [tuple(state_dict[k].shape) for k in keys]


def _actor_from(state_dict):
    """Actor matching the checkpoint exactly (any MLP variant)."""
    import torch.nn as nn
    from networks import device

    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_bound = 5.0
            self.net = _sequential_from_shapes(_linear_shapes(state_dict), nn.Tanh())

        def forward(self, s):
            return self.net(s) * self.action_bound

    return _Actor().to(device)


def _critic_from(state_dict):
    """Critic matching the checkpoint exactly (any MLP variant)."""
    import torch
    from networks import device

    class _Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _sequential_from_shapes(_linear_shapes(state_dict))

        def forward(self, s, a):
            return self.net(torch.cat([s, a], dim=-1))

    return _Critic().to(device)


def load_td3(path: str, freeze_attention: bool = False):
    """Load a checkpoint, rebuilding the network it actually contains."""
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    agent = payload['agent']
    arch = agent['arch']
    if arch == 'mlp':
        state_dim = agent['actor']['net.0.weight'].shape[1]
        ctrl = build_td3('mlp', state_dim, freeze_attention)
        ctrl.actor = _actor_from(agent['actor'])
        ctrl.critic1 = _critic_from(agent['critic1'])
        ctrl.critic2 = _critic_from(agent['critic2'])
        ctrl.target_actor = _actor_from(agent['target_actor'])
        ctrl.target_critic1 = _critic_from(agent['target_critic1'])
        ctrl.target_critic2 = _critic_from(agent['target_critic2'])
    else:
        state_dim = agent['actor']['gru_predictor.gru.weight_ih_l0'].shape[1]
        ctrl = build_td3('seq', state_dim, freeze_attention)
    ctrl.load_state(agent)
    return ctrl
