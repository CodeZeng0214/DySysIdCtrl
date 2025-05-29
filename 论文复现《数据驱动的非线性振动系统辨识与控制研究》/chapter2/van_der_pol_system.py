"""
Van der Pol振动系统模型
实现论文第2章中的Van der Pol振子动力学方程
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import rk4_step
from utils.math_utils import linearize_system

class VanDerPolSystem:
    """Van der Pol振动系统"""
    
    def __init__(self, m=1.0, c=1.0, k=100.0, l=1.0, e=0.05):
        """
        初始化Van der Pol系统参数
        
        参数:
        m: 质量 (kg)
        c: 阻尼系数 (kg/s) 
        k: 刚度 (kg/s²)
        l: 非线性参数 (kg·m⁻²·s⁻¹)
        e: 限制参数
        """
        self.m = m
        self.c = c
        self.k = k
        self.l = l
        self.e = e
        
        # 计算无量纲参数
        self.omega0 = np.sqrt(k/m)  # 固有频率
        self.zeta = c / (2 * np.sqrt(k*m))  # 阻尼比
        self.mu = l / (m * self.omega0)  # 非线性系数
        
        print(f"系统参数: m={m}, c={c}, k={k}, l={l}, e={e}")
        print(f"固有频率: {self.omega0:.3f} rad/s")
        print(f"阻尼比: {self.zeta:.3f}")
        print(f"非线性系数: {self.mu:.3f}")
    
    def dynamics(self, t, state, u=0, external_force=0):
        """
        Van der Pol系统动力学方程
        
        参数:
        t: 时间
        state: 状态向量 [x1, x2] = [位移, 速度]
        u: 控制力
        external_force: 外部激励力
        
        返回:
        状态导数 [dx1/dt, dx2/dt]
        """
        x1, x2 = state
        
        # Van der Pol方程: mẍ + c(1-e*x²)ẋ + kx = u + f_ext
        # 转换为状态空间形式:
        # ẋ1 = x2
        # ẋ2 = -(k/m)*x1 - (c/m)*(1-e*x1²)*x2 + (1/m)*(u + f_ext)
        
        dx1_dt = x2
        dx2_dt = (-(self.k/self.m)*x1 - 
                  (self.c/self.m)*(1 - self.e*x1**2)*x2 + 
                  (1/self.m)*(u + external_force))
        
        return np.array([dx1_dt, dx2_dt])
    
    def dynamics_normalized(self, t, state, u=0, external_force=0):
        """
        归一化的Van der Pol动力学方程
        """
        x1, x2 = state
        
        # 归一化方程: ẍ + μ(x²-1)ẋ + x = u_norm + f_norm
        dx1_dt = x2
        dx2_dt = -x1 + self.mu*(1 - self.e*x1**2)*x2 + u + external_force
        
        return np.array([dx1_dt, dx2_dt])
    
    def simulate(self, t_span, initial_state, control_func=None, external_force_func=None, dt=0.01):
        """
        仿真Van der Pol系统
        
        参数:
        t_span: 时间范围 [t_start, t_end]
        initial_state: 初始状态 [x1_0, x2_0]
        control_func: 控制函数 u(t, state)
        external_force_func: 外部激励函数 f(t)
        dt: 时间步长
        
        返回:
        t: 时间数组
        states: 状态历史
        controls: 控制力历史
        """
        t_eval = np.arange(t_span[0], t_span[1], dt)
        states = np.zeros((len(t_eval), 2))
        controls = np.zeros(len(t_eval))
        
        states[0] = initial_state
        
        for i in range(1, len(t_eval)):
            t_current = t_eval[i-1]
            state_current = states[i-1]
            
            # 计算控制力
            if control_func is not None:
                u = control_func(t_current, state_current)
            else:
                u = 0
            
            # 计算外部激励
            if external_force_func is not None:
                f_ext = external_force_func(t_current)
            else:
                f_ext = 0
            
            # 使用RK4积分
            states[i] = rk4_step(self.dynamics, t_current, state_current, dt, u, f_ext)
            controls[i-1] = u
        
        # 最后一个控制力
        if control_func is not None:
            controls[-1] = control_func(t_eval[-1], states[-1])
        
        return t_eval, states, controls
    
    def get_equilibrium_point(self):
        """获取平衡点"""
        return np.array([0.0, 0.0])
    
    def linearize_at_equilibrium(self):
        """在平衡点处线性化"""
        x_eq = self.get_equilibrium_point()
        u_eq = 0.0
        
        # 手动计算线性化矩阵
        A = np.array([[0, 1],
                      [-self.k/self.m, -self.c/self.m]])
        
        B = np.array([[0],
                      [1/self.m]])
        
        return A, B
    
    def compute_limit_cycle(self, num_points=1000):
        """计算极限环（近似）"""
        # 对于Van der Pol振子，极限环半径约为2
        theta = np.linspace(0, 2*np.pi, num_points)
        r = 2.0  # 近似半径
        
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        
        return x1, x2
    
    def is_in_safe_region(self, state, threshold=3.0):
        """判断状态是否在安全区域内"""
        x1, x2 = state
        return np.sqrt(x1**2 + x2**2) < threshold
    
    def plot_phase_portrait(self, initial_conditions=None, t_span=(0, 10), dt=0.01):
        """绘制相图"""
        if initial_conditions is None:
            initial_conditions = [[0.5, 0.5], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
        
        plt.figure(figsize=(10, 8))
        
        # 绘制向量场
        x1_range = np.linspace(-3, 3, 20)
        x2_range = np.linspace(-3, 3, 20)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        DX1 = X2
        DX2 = -X1 + self.mu*(1 - self.e*X1**2)*X2
        
        plt.quiver(X1, X2, DX1, DX2, alpha=0.5, color='gray')
        
        # 绘制轨迹
        colors = ['blue', 'red', 'green', 'orange']
        for i, ic in enumerate(initial_conditions):
            t, states, _ = self.simulate(t_span, ic, dt=dt)
            plt.plot(states[:, 0], states[:, 1], 
                    color=colors[i % len(colors)], linewidth=2,
                    label=f'IC: ({ic[0]}, {ic[1]})')
            
            # 标记起点
            plt.plot(ic[0], ic[1], 'o', color=colors[i % len(colors)], markersize=8)
        
        # 绘制极限环
        x1_lc, x2_lc = self.compute_limit_cycle()
        plt.plot(x1_lc, x2_lc, 'k--', linewidth=2, alpha=0.7, label='极限环')
        
        plt.xlabel('位移 x₁')
        plt.ylabel('速度 x₂')
        plt.title('Van der Pol振子相图')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.tight_layout()
        
        return plt.gcf()


class VanDerPolEnvironment:
    """Van der Pol强化学习环境"""
    
    def __init__(self, system, dt=0.01, episode_length=1000, 
                 Q=None, R=None, termination_threshold=3.0):
        """
        初始化环境
        
        参数:
        system: VanDerPolSystem实例
        dt: 时间步长
        episode_length: 每轮最大步数
        Q: 状态权重矩阵
        R: 控制权重矩阵
        termination_threshold: 终止阈值
        """
        self.system = system
        self.dt = dt
        self.episode_length = episode_length
        self.termination_threshold = termination_threshold
        
        # 默认权重矩阵
        if Q is None:
            Q = np.array([[10, 0], [0, 1]])
        if R is None:
            R = 0.01
            
        self.Q = Q
        self.R = R
        
        # 状态和动作空间
        self.state_dim = 2
        self.action_dim = 1
        self.action_bound = 10.0  # 控制力限制
        
        # 环境状态
        self.reset()
    
    def reset(self, initial_state=None):
        """重置环境"""
        if initial_state is None:
            # 随机初始状态
            self.state = np.array([0.0, np.random.choice([-1, 1])])
        else:
            self.state = np.array(initial_state)
        
        self.step_count = 0
        self.total_cost = 0.0
        
        return self.state.copy()
    
    def step(self, action):
        """执行一步"""
        # 限制动作范围
        action = np.clip(action, -self.action_bound, self.action_bound)
        
        # 计算即时奖励
        reward = self.compute_reward(self.state, action)
        
        # 更新状态
        next_state = rk4_step(self.system.dynamics, 
                             self.step_count * self.dt, 
                             self.state, self.dt, action)
        
        self.state = next_state
        self.step_count += 1
        self.total_cost += -reward  # 成本为负奖励
        
        # 检查终止条件
        done = self.is_terminated()
        
        # 如果发散，给予惩罚
        if done and self.step_count < self.episode_length:
            reward -= 100  # 发散惩罚
        
        info = {
            'step': self.step_count,
            'total_cost': self.total_cost,
            'is_safe': self.system.is_in_safe_region(self.state, self.termination_threshold)
        }
        
        return self.state.copy(), reward, done, info
    
    def compute_reward(self, state, action):
        """计算奖励函数"""
        # 基于二次型代价函数的奖励
        state_cost = state.T @ self.Q @ state
        action_cost = self.R * action**2
        
        # 转换为奖励（负代价）
        reward = -(state_cost + action_cost)
        
        return reward
    
    def is_terminated(self):
        """判断是否终止"""
        # 超出安全区域或达到最大步数
        if not self.system.is_in_safe_region(self.state, self.termination_threshold):
            return True
        if self.step_count >= self.episode_length:
            return True
        return False
    
    def get_state_action_value(self, state, action):
        """计算状态-动作价值（用于评估）"""
        return self.compute_reward(state, action)


if __name__ == "__main__":
    # 测试Van der Pol系统
    system = VanDerPolSystem()
    
    # 测试自由振动
    t_span = (0, 20)
    initial_state = [0.1, 0.1]
    
    t, states, controls = system.simulate(t_span, initial_state)
    
    # 绘制时域响应
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, states[:, 0], 'b-', linewidth=2)
    plt.ylabel('位移 (m)')
    plt.title('Van der Pol系统自由振动响应')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, states[:, 1], 'r-', linewidth=2)
    plt.ylabel('速度 (m/s)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(states[:, 0], states[:, 1], 'g-', linewidth=2)
    plt.xlabel('位移 (m)')
    plt.ylabel('速度 (m/s)')
    plt.title('相图')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 绘制相图
    system.plot_phase_portrait()
    plt.show()
    
    # 测试环境
    env = VanDerPolEnvironment(system)
    state = env.reset()
    print(f"初始状态: {state}")
    
    for i in range(5):
        action = np.random.uniform(-1, 1)  # 随机动作
        next_state, reward, done, info = env.step(action)
        print(f"步骤 {i+1}: 动作={action:.3f}, 奖励={reward:.3f}, 完成={done}")
        if done:
            break
