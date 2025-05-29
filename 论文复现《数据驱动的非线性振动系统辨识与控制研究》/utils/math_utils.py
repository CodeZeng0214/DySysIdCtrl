"""
数学计算工具模块
"""

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize
from scipy.integrate import solve_continuous_lyapunov

def solve_lqr(A, B, Q, R):
    """求解LQR控制器"""
    try:
        # 求解代数黎卡提方程
        P = solve_continuous_lyapunov(A.T, -Q - A.T @ np.linalg.inv(R) @ B.T @ B @ A)
        # 简化求解方法
        P = linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K, P
    except Exception as e:
        print(f"LQR求解失败: {e}")
        # 使用简化方法
        K = np.linalg.inv(R) @ B.T @ Q
        return K, Q

def linearize_system(func, x_eq, u_eq, delta=1e-6):
    """在平衡点处线性化非线性系统"""
    n = len(x_eq)
    m = len(u_eq) if isinstance(u_eq, (list, np.ndarray)) else 1
    
    # 计算A矩阵 (∂f/∂x)
    A = np.zeros((n, n))
    for i in range(n):
        x_plus = x_eq.copy()
        x_minus = x_eq.copy()
        x_plus[i] += delta
        x_minus[i] -= delta
        
        f_plus = func(0, x_plus, u_eq)
        f_minus = func(0, x_minus, u_eq)
        
        A[:, i] = (f_plus - f_minus) / (2 * delta)
    
    # 计算B矩阵 (∂f/∂u)
    if m == 1:
        u_plus = u_eq + delta
        u_minus = u_eq - delta
        f_plus = func(0, x_eq, u_plus)
        f_minus = func(0, x_eq, u_minus)
        B = (f_plus - f_minus) / (2 * delta)
        B = B.reshape(-1, 1)
    else:
        B = np.zeros((n, m))
        for j in range(m):
            u_plus = u_eq.copy()
            u_minus = u_eq.copy()
            u_plus[j] += delta
            u_minus[j] -= delta
            
            f_plus = func(0, x_eq, u_plus)
            f_minus = func(0, x_eq, u_minus)
            
            B[:, j] = (f_plus - f_minus) / (2 * delta)
    
    return A, B

def compute_eigenvalues(A):
    """计算矩阵特征值"""
    eigenvals = np.linalg.eigvals(A)
    return eigenvals

def is_stable(A):
    """判断系统是否稳定（所有特征值实部小于0）"""
    eigenvals = compute_eigenvalues(A)
    return np.all(np.real(eigenvals) < 0)

def compute_controllability_matrix(A, B):
    """计算可控性矩阵"""
    n = A.shape[0]
    C = B.copy()
    
    for i in range(1, n):
        C = np.hstack([C, np.linalg.matrix_power(A, i) @ B])
    
    return C

def is_controllable(A, B):
    """判断系统是否可控"""
    C = compute_controllability_matrix(A, B)
    rank = np.linalg.matrix_rank(C)
    return rank == A.shape[0]

def compute_observability_matrix(A, C):
    """计算可观性矩阵"""
    n = A.shape[0]
    O = C.copy()
    
    for i in range(1, n):
        O = np.vstack([O, C @ np.linalg.matrix_power(A, i)])
    
    return O

def is_observable(A, C):
    """判断系统是否可观"""
    O = compute_observability_matrix(A, C)
    rank = np.linalg.matrix_rank(O)
    return rank == A.shape[0]

def compute_quadratic_cost(x, u, Q, R):
    """计算二次型代价函数"""
    cost = x.T @ Q @ x + u.T @ R @ u
    return cost

def finite_difference_gradient(func, x, h=1e-8):
    """有限差分法计算梯度"""
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
    
    return grad

def finite_difference_jacobian(func, x, h=1e-8):
    """有限差分法计算雅可比矩阵"""
    f0 = func(x)
    n = len(x)
    m = len(f0)
    
    jac = np.zeros((m, n))
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        jac[:, i] = (func(x_plus) - func(x_minus)) / (2 * h)
    
    return jac

def compute_rms(signal):
    """计算信号均方根值"""
    return np.sqrt(np.mean(signal**2))

def compute_variance(signal):
    """计算信号方差"""
    return np.var(signal)

def compute_coefficient_of_variation(data):
    """计算变异系数"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    return std_val / mean_val if mean_val != 0 else 0

def soft_constraint_loss(estimated_params, expected_signs):
    """软约束损失函数"""
    loss = 0.0
    for param, expected_sign in zip(estimated_params, expected_signs):
        if expected_sign > 0 and param < 0:
            loss += np.abs(param)
        elif expected_sign < 0 and param > 0:
            loss += np.abs(param)
    return loss

def physics_loss(estimated_params, epsilon=1e-8):
    """物理损失函数（基于参数变异系数）"""
    if len(estimated_params) < 2:
        return 0.0
    
    mean_val = np.mean(estimated_params)
    std_val = np.std(estimated_params)
    cv = std_val / (np.abs(mean_val) + epsilon)
    return cv

def tanh_activation(x):
    """tanh激活函数"""
    return np.tanh(x)

def relu_activation(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def sigmoid_activation(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def gaussian_pdf(x, mu, sigma):
    """高斯概率密度函数"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def clip_action(action, action_bounds):
    """限制动作范围"""
    low, high = action_bounds
    return np.clip(action, low, high)

def normalize_angle(angle):
    """将角度归一化到[-π, π]范围"""
    return np.arctan2(np.sin(angle), np.cos(angle))

def exponential_decay(initial_value, decay_rate, step):
    """指数衰减"""
    return initial_value * np.exp(-decay_rate * step)

def linear_decay(initial_value, final_value, current_step, total_steps):
    """线性衰减"""
    if current_step >= total_steps:
        return final_value
    decay_ratio = current_step / total_steps
    return initial_value * (1 - decay_ratio) + final_value * decay_ratio

def compute_frequency_response(A, B, C, D, frequencies):
    """计算线性系统频率响应"""
    responses = []
    for omega in frequencies:
        s = 1j * omega
        G = C @ np.linalg.inv(s * np.eye(A.shape[0]) - A) @ B + D
        responses.append(G)
    return np.array(responses)

def polynomial_features(x, degree):
    """生成多项式特征"""
    features = [np.ones(x.shape[0])]  # 常数项
    for d in range(1, degree + 1):
        features.append(x ** d)
    return np.column_stack(features)

def moving_average_filter(data, window_size):
    """移动平均滤波"""
    filtered = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    return filtered
