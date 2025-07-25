## 二自由度电磁阻尼器系统仿真环境
from typing import Callable, Tuple, List
import numpy as np
from scipy.linalg import expm
import torch
from tqdm import tqdm
from ddpg_agent import DDPGAgent
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ElectromagneticDamperEnv:
    """二自由度电磁阻尼器系统仿真环境"""
    def __init__(self, A:np.ndarray, B:np.ndarray, C:np.ndarray, D:np.ndarray, E:np.ndarray, Ts:float=0.001, T:float=10, 
                 z_func:Callable=None, r_func:Callable=None, all_state0:np.ndarray=None, 
                 obs_indices: List[int] = None, x1_limit:float=None, 
                 use_time_noise:bool=False, time_noise_std:float=0.0001):
        """
        ## 初始化环境参数\n
        Xdot = Ax + Bu + E*z\n
        Y = Cx + Du\n
        
        参数:
        - obs_indices: 用于选择观测状态的索引列表。默认为[5]表示只观测x2_dot(主结构加速度)
                      例如: [2, 5]表示观测吸振器加速度和主结构加速度
        - use_time_noise: 是否在时间步长中加入噪声
        - time_noise_std: 时间步长噪声的标准差
        """
        # 设置观测索引
        self.A = A  # 连续状态转移矩阵
        self.B = B  # 连续控制传递矩阵
        self.C = C  # 连续输出状态转移矩阵
        self.D = D  # 连续控制传递矩阵 
        self.E = E  # 地基扰动转移矩阵
        self.z_func:Callable = z_func # 地基扰动函数
        self.r_func:Callable = r_func # 奖励函数
        self.x1_limit = x1_limit # 吸振器位移限制
        self.Ts = Ts  # 标准采样时间 
        self.T = T    # 仿真总时长
        self.use_time_noise = use_time_noise  # 是否使用时间噪声
        self.time_noise_std = time_noise_std  # 时间噪声标准差
        self.current_dt = Ts  # 当前实际时间步长
        self.time = 0.0  # 当前仿真时间
        self.all_state0 = all_state0 if all_state0 is not None else np.zeros(6)  # 初始状态
        self.all_states = np.zeros(6) # 完整状态 [x1, v1, a1, x2, v2, a2]
        self.set_observation_indices(obs_indices, log=False)  # 设置观测状态索引
        self.discretize_system() # 离散化系统
        self.reset()  # 初始化状态
        
    def discretize_system(self, Ts=None):
        """离散化连续状态空间系统"""
        if Ts is not None:
            self.Ts = Ts
        n = self.A.shape[0]  # 状态维度
        m1 = self.B.shape[1]  # 控制输入维度
        m2 = self.E.shape[1]  # 外部干扰维度
        
        # 构建扩展系统矩阵
        M = np.zeros((n + m1 + m2, n + m1 + m2))
        M[:n, :n] = self.A
        M[:n, n:n+m1] = self.B
        M[:n, n+m1:] = self.E
        
        # 计算矩阵指数
        expM = expm(M * self.Ts)
          # 提取离散化矩阵
        self.Ad = expM[:n, :n]  # 离散状态转移矩阵
        self.Bd = expM[:n, n:n+m1]  # 离散控制输入矩阵
        self.Ed = expM[:n, n+m1:]  # 离散外部干扰矩阵
        
    def get_current_timestep(self):
        """获取当前时间步长（可能包含噪声）"""
        if self.use_time_noise:
            # 添加高斯噪声到时间步长
            noise = np.random.normal(0, self.time_noise_std)
            # 限制时间步长在合理范围内（0.5*Ts 到 1.5*Ts）
            dt = np.clip(self.Ts + noise, 0.5 * self.Ts, 1.5 * self.Ts)
            self.current_dt = dt
            return dt
        else:
            self.current_dt = self.Ts
            return self.Ts
            
    def discretize_with_timestep(self, dt):
        """使用指定时间步长离散化系统"""
        n = self.A.shape[0]  # 状态维度
        m1 = self.B.shape[1]  # 控制输入维度
        m2 = self.E.shape[1]  # 外部干扰维度
        
        # 构建扩展系统矩阵
        M = np.zeros((n + m1 + m2, n + m1 + m2))
        M[:n, :n] = self.A
        M[:n, n:n+m1] = self.B
        M[:n, n+m1:] = self.E
        
        # 使用实际时间步长计算矩阵指数
        expM = expm(M * dt)
        
        # 提取离散化矩阵
        Ad_dt = expM[:n, :n]  # 离散状态转移矩阵
        Bd_dt = expM[:n, n:n+m1]  # 离散控制输入矩阵
        Ed_dt = expM[:n, n+m1:]  # 离散外部干扰矩阵
        
        return Ad_dt, Bd_dt, Ed_dt
        
    def reset(self, X0=None, z_func:Callable=None) -> np.ndarray:
        """重置环境到初始状态，返回初始观测值"""
        if X0:
            self.all_states = X0
            # 随机扰动初始状态（可选）
            # self.state[1] = np.random.uniform(-0.5, 0.5)  # 随机初始速度
        else:
            self.all_states = self.all_state0
            
        self.time = 0.0
        if z_func is not None:
            self.z_func = z_func  # 重置扰动函数
        return self.get_observation()  # 返回初始观测值
    
    def get_observation(self) -> np.ndarray:
        """获取当前时刻的观测值"""
        return np.array([self.all_states[i] for i in self.obs_indices]).copy()  # 返回选定的观测状态
        
    def get_Z(self, dt=None)-> np.ndarray:
        """获取扰动的速度和位移\n
        参数：\n
        - dt: 时间步长，如果为None则使用self.current_dt\n
        返回：Z = [[z_dot], [z]]"""
        if self.z_func is None:
            return np.zeros((2,1))  # 如果没有扰动函数，返回零矩阵
        
        if dt is None:
            dt = self.current_dt
            
        z_func = self.z_func
        z_dot = (z_func(self.time) - z_func(self.time - dt)) / dt  # 计算扰动的导数
        return np.array([[z_dot], [z_func(self.time)]])  # 返回扰动的速度和位移
        
    def set_observation_indices(self, obs_indices: List[int], log:bool=True):
        """设置观测状态的索引"""
            # Observation mapping for better logging
        state_names = {
            0: "吸振器位移 (x1)",
            1: "吸振器速度 (x1_dot)",
            2: "吸振器加速度 (x1_ddot)",
            3: "平台位移 (x2)",
            4: "平台速度 (x2_dot)",
            5: "平台加速度 (x2_ddot)"
        }
        obs_indices = obs_indices if obs_indices is not None else [3]  # 默认观测 x2
        observed_states = [state_names.get(idx, f"状态{idx}") for idx in obs_indices]
        if log: logging.info(f"观测量: {', '.join(observed_states)}")
        self.obs_indices = obs_indices
        self.obs_dim = len(obs_indices)
    
    def set_disturbance(self, z_func:Callable):
        """设置外部扰动函数"""
        self.z_func = z_func  # 设置扰动函数
        
    def set_reward_function(self, r_func:Callable):
        """设置奖励函数"""
        self.r_func = r_func  # 设置奖励函数
        
    def step(self, action:np.ndarray)-> Tuple[np.ndarray, float, bool, float]:
        """执行一个控制动作，更新系统状态，返回观测值、奖励、是否结束、时间步长等信息"""
        # 检查动作的类型
        if isinstance(action, torch.Tensor): action = action.cpu().numpy()
        elif isinstance(action, float): action = np.array([action])
        
        before_states = self.all_states.copy()  # 记录当前状态
        
        # 获取当前时间步长（可能包含噪声）
        dt = self.get_current_timestep()
        
        # 使用实际时间步长离散化系统
        Ad_dt, Bd_dt, Ed_dt = self.discretize_with_timestep(dt)
        
        # 应用离散状态方程更新内部状态
        # Xdot = Ax + Bu + E*z
        X = self.all_states[[0,1,3,4]].copy() # 提取 x 和 v
        next_X = Ad_dt @ X.reshape(-1, 1) + Bd_dt @ action.reshape(-1, 1) + Ed_dt @ self.get_Z(dt)
        
        # 限制吸振器位移
        if self.x1_limit and (abs(next_X[0]-next_X[2]) > self.x1_limit):
            next_X[0] = self.x1_limit * np.sign(next_X[0]-next_X[2]) + next_X[2]
        
        Y = self.C @ next_X.reshape(-1, 1) + self.D @ action.reshape(-1, 1)
        
        # 更新内部状态和时间
        self.all_states[[0,3]] = next_X[[0,2]].reshape(-1) # 更新位移
        self.all_states[[1,4]] = next_X[[1,3]].reshape(-1) # 更新速度
        self.all_states[[2,5]] = Y.reshape(-1) # 更新加速度

        self.time += dt  # 使用实际时间步长更新时间
        
        # 计算奖励
        if self.r_func is not None:
            reward:float = self.r_func(before_states, action, self.all_states.copy())
        else:
            reward = 0.0
        
        # 获得观测值
        observation = self.get_observation()
        
        # 判断是否结束
        done:bool = self.time >= self.T
        
        # 返回观测值、奖励、是否结束、时间步长
        return observation, reward, done, dt
    
    def run_simulation(self, controller=None, X0=None, z_func=None, r_func=None):
        """运行完整仿真"""
        
        # 重置环境并设置初始状态
        self.reset(X0)
        
        # 设置外部地基扰动
        if z_func is not None:
            self.set_disturbance(z_func)
        
        # 记录仿真数据
        full_states = [self.all_states.copy()] # 记录完整状态历史
        observations = [self.get_observation()] # 记录观测值历史
        rewards = [0.0] # 记录奖励历史
        actions = [0.0] # 记录动作历史
        times = [0.0]  # 记录时间历史
        dt_history = []  # 记录时间步长历史
        
        done = False
        tqdm_bar = tqdm(total=int(self.T/self.Ts), desc="仿真进度")
        while not done:
            obs = self.get_observation()  # 获取当前观测值
            if controller is not None: # DDPG 控制器
                action = controller.select_action(obs, add_noise=False,rand_prob=0)
            else:
                # 无控制
                action = 0.0
                
            # 执行一步仿真，获取下一个观测值、奖励等
            # step内部会更新 self.state
            next_obs, reward, done, dt = self.step(action)

            rewards.append(reward)  # 记录奖励
            dt_history.append(dt)  # 记录时间步长
            
            tqdm_bar.update(1)  # 更新进度条
            
            # 记录数据
            full_states.append(self.all_states.copy()) # 记录更新后的完整状态
            observations.append(next_obs)
            actions.append(action.item() if isinstance(action, np.ndarray) else action) # 记录动作
            times.append(self.time)
        
        return {
            'all_states': np.array(full_states), # 返回完整状态历史
            'observations': np.array(observations), # 返回观测值历史
            'actions': np.array(actions), # 返回动作历史
            'times': times, # 返回时间历史
            'rewards': rewards, # 返回奖励历史
            'dt_history': dt_history  # 返回时间步长历史
        }