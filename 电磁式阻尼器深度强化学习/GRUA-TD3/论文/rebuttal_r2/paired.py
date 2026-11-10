"""Paired evaluation: identical delay/jitter realisations for every controller.

The environment draws the loop delay and the sampling interval from the global
NumPy RNG, and the controller also draws from that same RNG (the exploration
noise is sampled even when its scale is zero).  The two streams therefore drift
apart, and the no-control baseline and the controlled run no longer see the same
disturbance realisation.

This module pre-generates the two sequences from a dedicated RandomState and
injects them into the environment, so that
  * every controller and the no-control baseline share the same realisation,
  * the comparison is paired, episode by episode.
"""
from __future__ import annotations

import numpy as np

import eval_common as E


def make_sequences(seed: int, n_steps: int, delay_mean: int, delay_std: int = 1,
                   Ts: float = 1e-3, dt_std: float = 0.1):
    """Replicates Env._sample_delay_steps then Env._sample_dt for a given seed."""
    rs = np.random.RandomState(seed)
    delays, dts = [], []
    for _ in range(n_steps):
        raw = np.clip(rs.normal(delay_mean, delay_std), 0, delay_mean * 2)
        delays.append(max(0, int(round(raw))))
        noise = rs.normal(0.0, dt_std * Ts)
        dts.append(float(np.clip(Ts + noise, 0.5 * Ts, 1.5 * Ts)))
    return iter(delays), iter(dts)


def attach(env, delays, dts, use_delay: bool):
    """Replace the two samplers of the environment by the injected sequences."""
    if use_delay:
        env._sample_delay_steps = lambda: next(delays)
    env._sample_dt = lambda: next(dts)


def rollout_paired(env, controller, seed, n_steps=4000, delay_mean=3,
                   mode="delayed"):
    """One episode with the injected realisation; returns the state history."""
    delays, dts = make_sequences(seed, n_steps, delay_mean)
    attach(env, delays, dts, use_delay=env.delay_enabled)
    return E.rollout(env, controller, mode=mode)
