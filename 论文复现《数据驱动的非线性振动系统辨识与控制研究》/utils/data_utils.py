"""
数据处理工具模块
"""

import numpy as np
import scipy.signal as signal
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

def generate_white_noise(duration, dt, psd=1e-4, seed=None):
    """生成高斯白噪声信号"""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(0, duration, dt)
    noise = np.random.normal(0, np.sqrt(psd), len(t))
    return t, noise

def generate_swept_sine(duration, dt, f_start=0.1, f_end=10, amplitude=1.0):
    """生成扫频正弦信号"""
    t = np.arange(0, duration, dt)
    freq_sweep = np.linspace(f_start, f_end, len(t))
    phase = 2 * np.pi * np.cumsum(freq_sweep) * dt
    signal = amplitude * np.sin(phase)
    return t, signal

def generate_chirp_signal(duration, dt, f0=0.5, f1=5.0, amplitude=1.0):
    """生成线性调频信号"""
    t = np.arange(0, duration, dt)
    signal_data = amplitude * signal.chirp(t, f0, duration, f1, method='linear')
    return t, signal_data

def generate_random_force(duration, dt, amplitude_range=(-5, 5), seed=None):
    """生成随机力信号"""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(0, duration, dt)
    force = np.random.uniform(amplitude_range[0], amplitude_range[1], len(t))
    return t, force

def compute_fft(signal_data, dt):
    """计算信号的FFT"""
    N = len(signal_data)
    freq = np.fft.fftfreq(N, dt)[:N//2]
    fft_data = np.fft.fft(signal_data)[:N//2]
    magnitude = np.abs(fft_data) / N
    phase = np.angle(fft_data)
    return freq, magnitude, phase

def compute_psd(signal_data, dt, nperseg=1024):
    """计算功率谱密度"""
    fs = 1.0 / dt
    freq, psd = signal.welch(signal_data, fs, nperseg=nperseg)
    return freq, psd

def compute_transmissibility(input_signal, output_signal, dt, nperseg=1024):
    """计算传递率"""
    fs = 1.0 / dt
    freq, Pxy = signal.csd(input_signal, output_signal, fs, nperseg=nperseg)
    _, Pxx = signal.welch(input_signal, fs, nperseg=nperseg)
    
    # 传递率 = |Pxy| / sqrt(Pxx * Pyy)
    transmissibility = np.abs(Pxy) / np.sqrt(Pxx * np.abs(Pxy))
    return freq, transmissibility

def rk4_step(func, t, y, dt, *args):
    """4阶龙格库塔单步求解"""
    k1 = dt * func(t, y, *args)
    k2 = dt * func(t + dt/2, y + k1/2, *args)
    k3 = dt * func(t + dt/2, y + k2/2, *args)
    k4 = dt * func(t + dt, y + k3, *args)
    
    y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y_next

def integrate_system(dynamics_func, y0, t_span, dt, *args):
    """使用RK4方法积分动力学系统"""
    t_eval = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(t_eval), len(y0)))
    y[0] = y0
    
    for i in range(1, len(t_eval)):
        y[i] = rk4_step(dynamics_func, t_eval[i-1], y[i-1], dt, *args)
    
    return t_eval, y

def normalize_data(data, method='minmax'):
    """数据归一化"""
    if method == 'minmax':
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        normalized = (data - data_min) / (data_max - data_min)
        return normalized, (data_min, data_max)
    elif method == 'zscore':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / std
        return normalized, (mean, std)
    else:
        raise ValueError("method must be 'minmax' or 'zscore'")

def denormalize_data(normalized_data, normalization_params, method='minmax'):
    """数据反归一化"""
    if method == 'minmax':
        data_min, data_max = normalization_params
        return normalized_data * (data_max - data_min) + data_min
    elif method == 'zscore':
        mean, std = normalization_params
        return normalized_data * std + mean
    else:
        raise ValueError("method must be 'minmax' or 'zscore'")

def create_dataset(X, y, sequence_length, overlap=0.5):
    """创建时间序列数据集"""
    step = int(sequence_length * (1 - overlap))
    X_seq = []
    y_seq = []
    
    for i in range(0, len(X) - sequence_length, step):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

def moving_average(data, window_size):
    """移动平均滤波"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def butter_filter(data, cutoff_freq, fs, filter_type='low', order=4):
    """巴特沃斯滤波器"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    
    if filter_type == 'low':
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'high':
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    elif filter_type == 'band':
        if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
            low = cutoff_freq[0] / nyquist
            high = cutoff_freq[1] / nyquist
            b, a = signal.butter(order, [low, high], btype='band', analog=False)
        else:
            raise ValueError("For bandpass filter, cutoff_freq must be [low, high]")
    else:
        raise ValueError("filter_type must be 'low', 'high', or 'band'")
    
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def calculate_mse(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)

def calculate_rmse(y_true, y_pred):
    """计算均方根误差"""
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_r_squared(y_true, y_pred):
    """计算决定系数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def sliding_window(data, window_size, step=1):
    """滑动窗口"""
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    return np.array(windows)
