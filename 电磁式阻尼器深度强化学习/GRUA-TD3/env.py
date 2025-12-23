from typing import Callable, List, Optional, Tuple
import numpy as np
from scipy.linalg import expm
import torch

from data import EpisodeRecorder
from controller import BaseController

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_env(ENV_PARAMS):
    # 系统参数
    m = 1.0    # 电磁吸振器质量
    M = 15  # 待减振对象质量
    k_m = 30_000  # 电磁吸振器刚度
    k_M = 300_000  # 平台刚度
    k_f = 100 # * TD3_PARAMS['action_bound']  # 电—力常数 N/A
    # k_E = 0.0  # 作动器反电动势系数
    # L = 0.0045  # 线圈的电感
    # R_m = 5.0  # 线圈的电阻
    c_m = 0.001 # 1.0  # 电磁吸振器阻尼
    c_M = 0.01 # 5.0  # 平台阻尼

    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-k_m / m, -c_m / m, k_m / m, c_m / m],
            [0.0, 0.0, 0.0, 1.0],
            [k_m / M, c_m / M, -(k_m + k_M) / M, -(c_m + c_M) / M],
        ]
    )
    B = np.array([[0.0], [k_f / m], [0.0], [-k_f / M]])
    C = np.array(
        [
            [-k_m / m, -c_m / m, k_m / m, c_m / m],
            [k_m / M, c_m / M, -(k_m + k_M) / M, -(c_m + c_M) / M],
        ]
    )
    D = np.array([[+k_f / m], [-k_f / M]])
    E = np.array([[0.0, 0.0, 0.0, c_M / M], [0.0, 0.0, 0.0, k_M / M]]).T
    F = np.array([[0.0], [0.0], [0.0], [1 / M]])

    env = ElectromagneticDamperEnv(A=A, B=B, C=C, D=D, E=E, F=F,
                                   Ts=ENV_PARAMS['Ts'], T=ENV_PARAMS['T'], 
                                   state0=ENV_PARAMS['state0'], obs_indices=ENV_PARAMS['obs_indices'], x1_limit=ENV_PARAMS['x1_limit'], 
                                   use_dt_noise=ENV_PARAMS['use_dt_noise'], dt_noise_std=ENV_PARAMS['dt_noise_std'], 
                                   delay_enabled=ENV_PARAMS['delay_enabled'], delay_mean_steps=ENV_PARAMS['delay_mean_steps'], delay_std_steps=ENV_PARAMS['delay_std_steps'], 
                                   include_dt_in_obs=ENV_PARAMS['include_dt_in_obs'], include_delay_in_obs=ENV_PARAMS['include_delay_in_obs'], 
                                   z_func=ENV_PARAMS['z_func'], r_func=ENV_PARAMS['r_func'], f_func=ENV_PARAMS['f_func'],
    )
    return env

class ElectromagneticDamperEnv:
    """Two-DOF electromagnetic damper simulation with optional delay and dt noise.
    二自由度电磁阻尼器仿真，具有可选的延迟和时间步长噪声。
    The environment owns all timing effects (dt noise, action delay) so the agent stays algorithmically clean.
    环境拥有所有时间效应（时间步长噪声，动作延迟），因此代理保持算法的纯净。
    """

    def __init__(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, E: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None,
        Ts: float = 1e-3, T: float = 1.0,
        state0: Optional[np.ndarray] = None, x1_limit: Optional[float] = None,
        use_dt_noise: bool = False, dt_noise_std: float = 0.0,
        delay_enabled: bool = False, delay_mean_steps: int = 1, delay_std_steps: int = 0,
        obs_indices: Optional[List[int]] = None, include_dt_in_obs: bool = False, include_delay_in_obs: bool = False,
        z_func: Optional[Callable] = None, r_func: Optional[Callable] = None, f_func: Optional[Callable] = None,
    ) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E if E is not None else np.zeros((A.shape[0], 2)) # 地基扰动输入矩阵
        self.F = F if F is not None else np.zeros((A.shape[0], 1)) # 外部激励输入矩阵

        self.Ts = Ts # 默认时间步长
        self.T = T # 总仿真时间
        self.time = 0.0 # 当前的仿真时间

        self.x1_limit = x1_limit # x1位置的限制

        self.use_dt_noise = use_dt_noise # 是否启用时间步长噪声
        self.dt_noise_std = dt_noise_std # 时间步长噪声的标准差比例
        self.dt_history = [] # 记录时间步长的历史

        self.delay_enabled = delay_enabled # 是否启用动作延迟
        self.delay_mean_steps = int(max(1, delay_mean_steps)) # 延迟的平均步数
        self.delay_std_steps = int(delay_std_steps) # 延迟的标准差步数
        self.delay_time = 0.0 # 当前的延迟时间
        self.delay_step: int = 0 # 当前的延迟步数

        self.obs_indices = obs_indices if obs_indices is not None else [3] # 观测变量的索引
        self.include_dt_in_obs = include_dt_in_obs # 是否在观测中包含时间步长
        self.include_delay_in_obs = include_delay_in_obs # 是否在观测中包含延迟时间

        self.state0 = state0 if state0 is not None else np.zeros(6) # 环境的初始状态
        self.z_func = z_func # 地基扰动函数
        self.f_func = f_func # 外部激励函数
        self.r_func = r_func # 奖励函数

        self._state = self.state0.copy() # 当前的状态
        self.state_history: List[np.ndarray] = [] # 按时间顺序保存历史状态，用于延迟观测对齐
        self._precompute_discrete(self.Ts) # 预计算离散时间系统矩阵
 

    # ------------------------------------------------------------------
    # Public API
    def reset(self, state0: Optional[np.ndarray] = None, z_func: Optional[Callable] = None, f_func: Optional[Callable] = None) -> np.ndarray:
        """环境初始化，\n
        可以重新设置初始状态和函数"""
        # 重新设置初始状态和函数
        if state0 is not None: 
            self.state0 = state0.copy()
        if z_func is not None: 
            self.z_func = z_func
        if f_func is not None: 
            self.f_func = f_func

        # 重置环境状态
        self.time = 0.0
        self.dt_history = []
        self.delay_time = 0.0
        self.delay_step: int = 0
        self._state = self.state0.copy()
        self.state_history = [self._state.copy()]

        return self.observe()

    def observe(self) -> np.ndarray:
        """获取当前观测值，包含指定的状态变量和可选的时间步长及延迟时间\n
        如果有时延，则返回历史中的延迟状态（前 delay_step 步）。"""
        # 计算延迟对齐的状态索引
        idx = max(0, len(self.state_history) - self.delay_step - 1 )
        state_ref = self.state_history[idx]

        base = np.array([state_ref[i] for i in self.obs_indices], dtype=float)
        extras = []
        if self.include_dt_in_obs:
            extras.append(self.dt_history[-1] if self.dt_history else self.Ts)
        if self.include_delay_in_obs:
            extras.append(self.delay_time)
        if extras:
            base = np.concatenate([base, extras])
        return base
    
    def get_all_state(self) -> np.ndarray:
        """获取完整的环境状态向量"""
        return self._state.copy()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """单步推进，不再延迟动作，延迟只体现在观测对齐。\n
        info 中包含每一步采样的时间步长和延迟步数。"""
        if self.delay_enabled:
            self.delay_step = self._sample_delay_steps() # 采样延迟步数
            self.delay_time = self._cal_delay_time(self.delay_step) # 计算对应的延迟时间
        
        # 如果启用时间步长噪声，则采样新的时间步长并重新离散系统矩阵
        dt = self.Ts
        if self.use_dt_noise:
            dt = self._sample_dt()
            self._precompute_discrete(dt)
        self.dt_history.append(dt)

        # 推进系统状态
        state = self._state.copy()
        next_state = self._integrate(action, dt)
        self._state = next_state
        self.state_history.append(next_state.copy())

        # 计算奖励
        reward = self.r_func(state, action, next_state) if self.r_func else 0.0
        next_obs = self.observe()

        # 注意，info 中包含当前的观测值、状态、时间、采样的时间步长和延迟步数、下一个状态
        info = {"state": state, "time": self.time, "next_state": next_state, "dt": dt, "delay_step": self.delay_step, "delay_time": self.delay_time}
        
        # 更新仿真时间
        self.time += dt
        done = self.time >= self.T
        
        return next_obs, reward, done, info

    def run_episode(self, state0: Optional[np.ndarray] = None,
                    z_func: Optional[Callable] = None, f_func: Optional[Callable] = None,
                    controller: BaseController=None, record: bool = True) -> EpisodeRecorder:
        """运行完整的仿真过程，直到达到总时间T。\n
        可以重新设置初始状态和函数，并使用指定的控制器进行控制。\n"""
        # 创建记录器
        if record:
            recorder = EpisodeRecorder() 
        
        # 重置控制器状态
        if controller is not None and hasattr(controller, "reset"):
            controller.reset()

        # 仿真主循环
        self.reset(state0=state0, z_func=z_func, f_func=f_func) # 重置环境
        done = False
        while not done:
            obs = self.observe() # 获取当前观测值
            action = controller.select_action(obs=obs) if controller is not None else 0.0 # 选择动作
            next_obs, reward, done, info = self.step(action) # 环境步进
            
            # 记录步进前观测、步进前状态、采取动作、获得奖励、步进前时间、步进时间步长、延迟时间
            if record:
                recorder.append(obs_history=obs.copy(), state_history=info["state"], action_history=action, reward_history=reward, 
                                time_history=info["time"], dt_history=info["dt"],  delay_time=info["delay_time"])
        return recorder

    # ------------------------------------------------------------------
    # Internal helpers
    def get_delay_state_seq(self, state_window: List[np.ndarray], delay_steps: int, seq_len: int) -> np.ndarray:
        """根据给定的延迟步数和序列长度，从状态窗口中提取对应的延迟状态序列"""
        assert len(state_window) >= delay_steps + seq_len, "状态窗口长度不足以提取所需的延迟序列"
        start_idx = len(state_window) - delay_steps - seq_len
        end_idx = len(state_window) - delay_steps
        seq = np.array(state_window[start_idx:end_idx], dtype=float)
        return seq

    def _integrate(self, action: float, dt: float) -> np.ndarray:
        """使用离散系统矩阵推进状态"""
        # 提取系统状态向量和计算地基扰动及外部激励
        X = self._state[[0, 1, 3, 4]].copy() # 系统状态向量：[x1, x1_dot, x2, x2_dot]
        z = self._get_ground_motion(dt) # 地基扰动向量：[z_dot, z]
        f = self._get_force() # 外部激励向量：[f]

        # 计算下一个状态
        X_next = self._Ad @ X.reshape(-1, 1) + self._Bd @ np.array([[action]]) + self._Ed @ z + self._Fd @ f 

        # 位置限制，防止吸振器位移发散
        if self.x1_limit is not None and abs(X_next[0] - X_next[2]) > self.x1_limit:
            X_next[0] = self.x1_limit * np.sign(X_next[0] - X_next[2]) + X_next[2]
            X_next[1] = 0.0

        # 计算输出变量
        Y = self.C @ X_next + self.D @ np.array([[action]])

        # 更新完整状态向量
        next_state = self._state.copy()
        next_state[[0, 3]] = X_next[[0, 2]].reshape(-1)
        next_state[[1, 4]] = X_next[[1, 3]].reshape(-1)
        next_state[[2, 5]] = Y.reshape(-1)
        return next_state
    
    def _cal_delay_time(self, delay_steps: int) -> float:
        """计算给定延迟步数对应的延迟时间"""
        return sum(self.dt_history[-delay_steps:])

    def _sample_dt(self) -> float:
        if not self.use_dt_noise:
            return self.Ts
        noise = np.random.normal(0.0, self.dt_noise_std * self.Ts)
        dt = float(np.clip(self.Ts + noise, 0.5 * self.Ts, 1.5 * self.Ts))
        return dt

    def _sample_delay_steps(self) -> int:
        """从正态分布中采样延迟步数"""
        raw = np.random.normal(self.delay_mean_steps, self.delay_std_steps)
        return max(0, int(round(raw)))

    def _precompute_discrete(self, dt: float) -> None:
        """离散化系统矩阵"""
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
        self._Ad = expM[:n, :n]
        self._Bd = expM[:n, n:n + m1]
        self._Ed = expM[:n, n + m1:n + m1 + m2]
        self._Fd = expM[:n, n + m1 + m2:]

    def _get_ground_motion(self, dt: float) -> np.ndarray:
        """计算地基扰动向量：[z_dot, z]"""
        if self.z_func is None:
            return np.zeros((2, 1))
        z_t = self.z_func(self.time)
        z_t_prev = self.z_func(self.time - dt)
        z_dot = (z_t - z_t_prev) / dt
        return np.array([[z_dot], [z_t]], dtype=float)

    def _get_force(self) -> np.ndarray:
        """计算外部激励向量：[f]"""
        if self.f_func is None:
            return np.zeros((1, 1))
        return np.array([[self.f_func(self.time)]], dtype=float)
