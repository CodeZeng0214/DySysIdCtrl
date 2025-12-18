from typing import Callable, List, Optional, Tuple
import numpy as np
from scipy.linalg import expm
import torch

from data import EpisodeRecorder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ElectromagneticDamperEnv:
    """Two-DOF electromagnetic damper simulation with optional delay and dt noise.

    The environment owns all timing effects (dt noise, action delay) so the agent stays algorithmically clean.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        E: np.ndarray,
        F: Optional[np.ndarray] = None,
        Ts: float = 1e-3,
        T: float = 1.0,
        state0: Optional[np.ndarray] = None,
        obs_indices: Optional[List[int]] = None,
        x1_limit: Optional[float] = None,
        use_dt_noise: bool = False,
        dt_noise_std: float = 0.0,
        delay_enabled: bool = False,
        delay_mean_steps: int = 1,
        delay_std_steps: float = 0.0,
        include_dt_in_obs: bool = False,
        include_delay_in_obs: bool = False,
        z_func: Optional[Callable[[float], float]] = None,
        r_func: Optional[Callable[[np.ndarray, float, np.ndarray], float]] = None,
        f_func: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F if F is not None else np.zeros((A.shape[0], 1))

        self.Ts = Ts
        self.T = T
        self.state0 = state0 if state0 is not None else np.zeros(6)
        self.obs_indices = obs_indices if obs_indices is not None else [3]
        self.x1_limit = x1_limit

        self.use_dt_noise = use_dt_noise
        self.dt_noise_std = dt_noise_std

        self.delay_enabled = delay_enabled
        self.delay_mean_steps = max(1, delay_mean_steps)
        self.delay_std_steps = delay_std_steps
        self.include_dt_in_obs = include_dt_in_obs
        self.include_delay_in_obs = include_delay_in_obs

        self.z_func = z_func
        self.r_func = r_func
        self.f_func = f_func

        self.all_state = self.state0.copy()
        self.time = 0.0
        self._precompute_discrete(self.Ts)
        self._init_delay_buffers(self.delay_mean_steps)

    # ------------------------------------------------------------------
    # Public API
    def reset(self, state0: Optional[np.ndarray] = None) -> np.ndarray:
        self.all_state = self.state0.copy() if state0 is None else state0.copy()
        self.time = 0.0
        delay_len = self.delay_mean_steps if self.delay_enabled else 1
        self._init_delay_buffers(delay_len)
        return self.observe()

    def observe(self) -> np.ndarray:
        base = np.array([self.all_state[i] for i in self.obs_indices], dtype=float)
        extras = []
        if self.include_dt_in_obs:
            extras.append(self.last_dt)
        if self.include_delay_in_obs:
            extras.append(self.delay_time)
        if extras:
            base = np.concatenate([base, np.array(extras, dtype=float)])
        return base

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        applied_action = self._apply_delay(action)
        dt = self._sample_dt()
        self._precompute_discrete(dt)

        before = self.all_state.copy()
        next_state = self._integrate(applied_action, dt)
        self.all_state = next_state
        self.time += dt

        reward = self.r_func(before, applied_action, next_state) if self.r_func else 0.0
        obs = self.observe()
        done = self.time >= self.T
        info = {"dt": dt, "delay_time": self.delay_time, "applied_action": applied_action}
        return obs, reward, done, info

    def run_episode(self, policy=None, record: bool = True) -> EpisodeRecorder:
        recorder = EpisodeRecorder()
        self.reset()
        if record:
            recorder.append(state=self.all_state.copy(), action=0.0, reward=0.0, dt=self.last_dt, time=self.time, delay_time=self.delay_time)
        done = False
        while not done:
            obs = self.observe()
            act = float(policy(obs)) if policy is not None else 0.0
            obs_next, reward, done, info = self.step(act)
            if record:
                recorder.append(state=self.all_state.copy(), action=act, reward=reward, dt=info["dt"], time=self.time, delay_time=info["delay_time"])
        return recorder

    # ------------------------------------------------------------------
    # Internal helpers
    def _integrate(self, action: float, dt: float) -> np.ndarray:
        X = self.all_state[[0, 1, 3, 4]].copy()
        z = self._get_ground_motion(dt)
        f = self._get_force()
        X_next = self.Ad @ X.reshape(-1, 1) + self.Bd @ np.array([[action]]) + self.Ed @ z + self.Fd @ f

        if self.x1_limit is not None and abs(X_next[0] - X_next[2]) > self.x1_limit:
            X_next[0] = self.x1_limit * np.sign(X_next[0] - X_next[2]) + X_next[2]
            X_next[1] = 0.0

        Y = self.C @ X_next + self.D @ np.array([[action]])

        next_state = self.all_state.copy()
        next_state[[0, 3]] = X_next[[0, 2]].reshape(-1)
        next_state[[1, 4]] = X_next[[1, 3]].reshape(-1)
        next_state[[2, 5]] = Y.reshape(-1)
        return next_state

    def _apply_delay(self, action: float) -> float:
        if not self.delay_enabled:
            self.last_dt = self.Ts
            self.delay_time = 0.0
            return float(action)

        desired_len = self._sample_delay_steps()
        self._resize_delay_buffers(desired_len)
        self.action_queue.append(float(action))
        applied_action = self.action_queue.pop(0)

        # Update delay time estimate using current dt
        self.delay_time = self.last_dt * max(0, len(self.action_queue))
        return float(applied_action)

    def _sample_dt(self) -> float:
        if not self.use_dt_noise:
            self.last_dt = self.Ts
            return self.Ts
        noise = np.random.normal(0.0, self.dt_noise_std * self.Ts)
        dt = float(np.clip(self.Ts + noise, 0.5 * self.Ts, 1.5 * self.Ts))
        self.last_dt = dt
        return dt

    def _sample_delay_steps(self) -> int:
        raw = np.random.normal(self.delay_mean_steps, self.delay_std_steps)
        return max(1, int(round(raw)))

    def _precompute_discrete(self, dt: float) -> None:
        n = self.A.shape[0]
        m1 = self.B.shape[1]
        m2 = self.E.shape[1]
        m3 = self.F.shape[1]
        M = np.zeros((n + m1 + m2 + m3, n + m1 + m2 + m3))
        M[:n, :n] = self.A
        M[:n, n:n + m1] = self.B
        M[:n, n + m1:n + m1 + m2] = self.E
        M[:n, n + m1 + m2:] = self.F
        expM = expm(M * dt)
        self.Ad = expM[:n, :n]
        self.Bd = expM[:n, n:n + m1]
        self.Ed = expM[:n, n + m1:n + m1 + m2]
        self.Fd = expM[:n, n + m1 + m2:]

    def _init_delay_buffers(self, delay_len: int) -> None:
        self.action_queue: List[float] = [0.0 for _ in range(max(1, delay_len))]
        self.last_dt = self.Ts
        self.delay_time = 0.0

    def _resize_delay_buffers(self, desired_len: int) -> None:
        desired_len = max(1, desired_len)
        if desired_len > len(self.action_queue):
            padding = [self.action_queue[0]] * (desired_len - len(self.action_queue))
            self.action_queue = padding + self.action_queue
        elif desired_len < len(self.action_queue):
            self.action_queue = self.action_queue[-desired_len:]

    def _get_ground_motion(self, dt: float) -> np.ndarray:
        if self.z_func is None:
            return np.zeros((2, 1))
        z_t = self.z_func(self.time)
        z_t_prev = self.z_func(self.time - dt)
        z_dot = (z_t - z_t_prev) / dt
        return np.array([[z_dot], [z_t]], dtype=float)

    def _get_force(self) -> np.ndarray:
        if self.f_func is None:
            return np.zeros((1, 1))
        return np.array([[self.f_func(self.time)]], dtype=float)
