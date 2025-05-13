## 二自由度电磁阻尼器系统仿真环境
from typing import Callable, Tuple
import numpy as np
from scipy.linalg import expm
import torch
from tqdm import tqdm
from ddpg_agent import DDPGAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ElectromagneticDamperEnv:
    """二自由度电磁阻尼器系统仿真环境"""
    def __init__(self, A:np.ndarray, B:np.ndarray, C:np.ndarray, D:np.ndarray, E:np.ndarray, Ts:float=0.001, T:float=10, z_func:Callable=None):
        """
        ## 初始化环境参数\n
        Xdot = Ax + Bu + E*z\n
        Y = Cx + Du\n
        """
        self.reset()  # 初始化状态
        self.A = A  # 连续状态转移矩阵
        self.B = B  # 连续控制传递矩阵
        self.C = C  # 连续输出状态转移矩阵
        self.D = D  # 连续控制传递矩阵 
        self.E = E  # 地基扰动转移矩阵
        self.z_func:Callable = z_func # 地基扰动函数
        self.Ts = Ts  # 采样时间
        self.T = T    # 仿真总时长
        self.time = 0.0  # 当前仿真时间
        self.state = np.zeros(4)  # 内部完整状态 [x1, x1_dot, x2, x2_dot]
        
        self.discretize_system() # 离散化系统
        
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
        
    def reset(self, X0=None, z_func:Callable=None)-> np.ndarray:
        """重置环境到初始状态，返回初始观测值"""
        if X0 is None:
            self.state = np.zeros(4)
            # 随机扰动初始状态（可选）
            # self.state[1] = np.random.uniform(-0.5, 0.5)  # 随机初始速度
        else:
            self.state = np.array(X0)
        self.time = 0.0
        if z_func:self.z_func = z_func  # 重置扰动函数
        return np.array([self.state[3]])
    
    def get_observation(self)-> np.ndarray:
        """获取当前时刻的观测值"""
        return np.array([self.state[3]])  # 返回主结构的加速度 (x2_dot)
        
    def get_Z(self)-> np.ndarray:
        """获取扰动的速度和位移\n
        返回：Z = [[z_dot], [z]]"""
        if self.z_func is None:
            return np.zeros((2,1))  # 如果没有扰动函数，返回零矩阵
        z_func = self.z_func # 获取当前时间的扰动值
        z_dot = (z_func(self.time) - z_func(self.time - self.Ts)) / self.Ts  # 计算扰动的导数
        return np.array([[z_dot], [z_func(self.time)]])  # 返回扰动的速度和位移
               
    def set_disturbance(self, z_func:Callable):
        """设置外部扰动函数"""
        self.z_func = z_func  # 设置扰动函数
        
    def step(self, action:np.ndarray)-> Tuple[np.ndarray, bool]:
        """执行一个控制动作，更新系统状态，返回观测值、是否结束等信息"""
        # 检查动作的类型
        if isinstance(action, torch.Tensor): action = action.cpu().numpy()
        elif isinstance(action, float): action = np.array([action])
            
        # 应用离散状态方程更新内部状态
        # Xdot = Ax + Bu + E*z
        next_state_full = self.Ad @ self.state.reshape(-1, 1) + self.Bd @ action.reshape(-1, 1) + self.Ed @ self.get_Z()
        
        # 更新内部状态和时间
        self.state = next_state_full.reshape(-1)  # 将状态转换为一维数组
        self.time += self.Ts
        
        # 获得观测值
        observation = self.get_observation()
        
        # 判断是否结束
        done:bool = self.time >= self.T
        
        # 返回观测值、是否结束
        return observation, done
    
    def run_simulation(self, controller:DDPGAgent=None, X0=None, z_func=None, r_func=None):
        """运行完整仿真"""
        
        # 重置环境并设置初始状态
        self.reset(X0)
        
        # 设置外部地基扰动
        if z_func is not None:
            self.set_disturbance(z_func)
        
        # 记录仿真数据
        full_states = [] # 记录完整状态历史
        observations = [] # 记录观测值历史
        rewards = [] # 记录奖励历史
        actions = []
        times = []
        
        done = False
        tqdm_bar = tqdm(total=int(self.T/self.Ts), desc="仿真进度")
        while not done:
            obs = self.get_observation()  # 获取当前观测值
            if controller is not None: # DDPG 控制器
                action = controller.select_action(obs, add_noise=False)
            else:
                # 无控制
                action = 0.0
                
            # 执行一步仿真，获取下一个观测值、奖励等
            # step内部会更新 self.state
            next_obs, done = self.step(action)
            
            reward = r_func(obs, action, next_obs) if r_func is not None else 0.0
            rewards.append(reward)  # 记录奖励
            
            tqdm_bar.update(1)  # 更新进度条
            
            # 记录数据
            full_states.append(self.state) # 记录更新后的完整状态
            observations.append(next_obs)
            actions.append(action)
            times.append(self.time)
            
            # 更新当前观测值，用于下一轮决策
            obs = next_obs
        
        return {
            'states': np.array(full_states), # 返回完整状态历史
            'observations': np.array(observations), # 返回观测值历史
            'actions': np.array(actions), # 返回动作历史
            'times': np.array(times), # 返回时间历史
            'rewards': np.array(rewards) # 返回奖励历史
        }