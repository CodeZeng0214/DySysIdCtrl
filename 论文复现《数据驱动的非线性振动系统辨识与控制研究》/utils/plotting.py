"""
绘图工具模块
用于生成论文中的各种图表
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.patches as patches

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def plot_phase_portrait(x1, x2, title="相图", xlabel="位移 (m)", ylabel="速度 (m/s)"):
    """绘制相图"""
    plt.figure(figsize=(8, 6))
    plt.plot(x1, x2, 'b-', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_time_response(t, x, u=None, title="时域响应"):
    """绘制时域响应"""
    if u is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        
        # 位移响应
        ax1.plot(t, x[:, 0], 'b-', linewidth=2, label='位移')
        ax1.set_ylabel('位移 (m)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 速度响应
        ax2.plot(t, x[:, 1], 'r-', linewidth=2, label='速度')
        ax2.set_ylabel('速度 (m/s)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 控制力
        ax3.plot(t, u, 'g-', linewidth=2, label='控制力')
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('控制力 (N)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # 位移响应
        ax1.plot(t, x[:, 0], 'b-', linewidth=2, label='位移')
        ax1.set_ylabel('位移 (m)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 速度响应
        ax2.plot(t, x[:, 1], 'r-', linewidth=2, label='速度')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('速度 (m/s)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_frequency_response(freq, mag, phase=None, title="频率响应"):
    """绘制频率响应"""
    if phase is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # 幅频特性
        ax1.semilogx(freq, 20*np.log10(mag), 'b-', linewidth=2)
        ax1.set_ylabel('幅值 (dB)')
        ax1.grid(True, alpha=0.3)
        
        # 相频特性
        ax2.semilogx(freq, phase, 'r-', linewidth=2)
        ax2.set_xlabel('频率 (Hz)')
        ax2.set_ylabel('相位 (度)')
        ax2.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogx(freq, 20*np.log10(mag), 'b-', linewidth=2)
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('幅值 (dB)')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_transmissibility(freq, trans, title="位移传递率"):
    """绘制传递率曲线"""
    plt.figure(figsize=(10, 6))
    plt.loglog(freq, trans, 'b-', linewidth=2)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='传递率=1')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('位移传递率')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def plot_training_curve(episodes, rewards, title="训练曲线", ylabel="累积奖励"):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, 'b-', linewidth=2)
    plt.xlabel('训练轮次')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_control_cost_comparison(controllers, costs, title="控制消耗比较"):
    """绘制控制消耗比较图"""
    plt.figure(figsize=(8, 6))
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars = plt.bar(controllers, costs, color=colors[:len(controllers)])
    
    # 添加数值标签
    for bar, cost in zip(bars, costs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(costs),
                f'{cost:.3f}', ha='center', va='bottom')
    
    plt.ylabel('控制消耗')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return plt.gcf()

def plot_van_der_pol_limit_cycle():
    """绘制Van der Pol振子的极限环"""
    # 生成极限环数据
    theta = np.linspace(0, 2*np.pi, 1000)
    r = 2
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x1, x2, 'r-', linewidth=3, label='极限环')
    
    # 添加向量场
    x1_grid = np.linspace(-3, 3, 15)
    x2_grid = np.linspace(-3, 3, 15)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Van der Pol方程的向量场 (μ=1)
    dX1 = X2
    dX2 = -X1 + (1 - X1**2) * X2
    
    ax.quiver(X1, X2, dX1, dX2, alpha=0.5, color='gray')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('位移 x₁')
    ax.set_ylabel('速度 x₂')
    ax.set_title('Van der Pol振子极限环')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_qzs_system_diagram():
    """绘制QZS隔振器结构示意图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制基础
    base = patches.Rectangle((-1, -0.2), 2, 0.2, linewidth=2, 
                           edgecolor='black', facecolor='gray', alpha=0.7)
    ax.add_patch(base)
    
    # 绘制质量块
    mass = patches.Rectangle((-0.5, 1.5), 1, 0.5, linewidth=2,
                           edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(mass)
    
    # 绘制竖直弹簧
    x_spring = np.linspace(0, 0, 10)
    y_spring = np.linspace(0, 1.5, 10)
    spring_x = 0.1 * np.sin(10 * np.pi * np.linspace(0, 1, 10))
    ax.plot(spring_x, y_spring, 'k-', linewidth=2)
    
    # 绘制水平弹簧
    # 左侧弹簧
    x_left = np.linspace(-0.8, -0.5, 10)
    y_left = np.ones(10) * 1.75
    spring_y_left = 1.75 + 0.05 * np.sin(10 * np.pi * np.linspace(0, 1, 10))
    ax.plot(x_left, spring_y_left, 'k-', linewidth=2)
    
    # 右侧弹簧
    x_right = np.linspace(0.5, 0.8, 10)
    y_right = np.ones(10) * 1.75
    spring_y_right = 1.75 + 0.05 * np.sin(10 * np.pi * np.linspace(0, 1, 10))
    ax.plot(x_right, spring_y_right, 'k-', linewidth=2)
    
    # 绘制支撑点
    ax.plot(-0.8, 0, 'ko', markersize=8)
    ax.plot(0.8, 0, 'ko', markersize=8)
    
    # 绘制阻尼器
    ax.plot([0.3, 0.3], [0, 1.5], 'r-', linewidth=3, label='阻尼器')
    
    # 添加标注
    ax.text(0, 1.75, 'm', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(0.15, 0.75, 'k', ha='center', va='center', fontsize=12)
    ax.text(-0.65, 1.9, 'k_h', ha='center', va='center', fontsize=12)
    ax.text(0.65, 1.9, 'k_h', ha='center', va='center', fontsize=12)
    ax.text(0.4, 0.75, 'c', ha='center', va='center', fontsize=12, color='red')
    
    # 添加坐标轴
    ax.arrow(0, 2.5, 0, 0.3, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax.text(0.1, 2.7, 'x', fontsize=12)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('准零刚度(QZS)隔振器结构示意图', fontsize=14, weight='bold')
    
    plt.tight_layout()
    return fig

def plot_multilayer_system_diagram():
    """绘制多级隔振系统示意图"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 系统参数
    levels = 3
    level_height = 2
    mass_width = 1
    mass_height = 0.4
    
    for i in range(levels + 1):
        y_pos = i * level_height
        
        if i == 0:
            # 基础激励
            base = patches.Rectangle((-mass_width/2, y_pos), mass_width, mass_height, 
                                   linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(base)
            ax.text(0, y_pos + mass_height/2, '基础激励', ha='center', va='center', fontsize=10)
        else:
            # 质量块
            mass = patches.Rectangle((-mass_width/2, y_pos), mass_width, mass_height,
                                   linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.7)
            ax.add_patch(mass)
            ax.text(0, y_pos + mass_height/2, f'm_{i}', ha='center', va='center', fontsize=12)
            
            # 弹簧和阻尼器
            if i < levels:
                # 线性弹簧
                spring_x = 0.2
                spring_y = np.linspace(y_pos + mass_height, (i+1)*level_height, 10)
                spring_offset = 0.05 * np.sin(10 * np.pi * np.linspace(0, 1, 10))
                ax.plot(spring_x + spring_offset, spring_y, 'k-', linewidth=2)
                ax.text(spring_x + 0.15, y_pos + level_height/2, f'k_{i}', fontsize=10)
                
                # 非线性弹簧
                nl_spring_x = -0.2
                nl_spring_y = np.linspace(y_pos + mass_height, (i+1)*level_height, 10)
                nl_spring_offset = 0.08 * np.sin(15 * np.pi * np.linspace(0, 1, 10))
                ax.plot(nl_spring_x + nl_spring_offset, nl_spring_y, 'r-', linewidth=2)
                ax.text(nl_spring_x - 0.25, y_pos + level_height/2, f'k_{i}x³', fontsize=10, color='red')
                
                # 阻尼器
                damp_x = 0
                damp_y = np.linspace(y_pos + mass_height, (i+1)*level_height, 2)
                ax.plot([damp_x, damp_x], damp_y, 'b-', linewidth=3)
                ax.text(damp_x + 0.1, y_pos + level_height/2, f'c_{i}', fontsize=10, color='blue')
    
    # 控制力箭头
    control_y = levels * level_height + mass_height/2
    ax.arrow(mass_width/2 + 0.3, control_y, 0.3, 0, head_width=0.1, 
             head_length=0.1, fc='green', ec='green', linewidth=2)
    ax.text(mass_width/2 + 0.5, control_y + 0.2, 'u(t)', fontsize=12, color='green')
    
    # 外部激励箭头
    excitation_y = 0 + mass_height/2
    ax.arrow(-mass_width/2 - 0.6, excitation_y, 0.3, 0, head_width=0.1,
             head_length=0.1, fc='orange', ec='orange', linewidth=2)
    ax.text(-mass_width/2 - 0.8, excitation_y + 0.2, 'f(t)', fontsize=12, color='orange')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, (levels + 1) * level_height)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('多级非线性隔振系统示意图', fontsize=14, weight='bold')
    
    plt.tight_layout()
    return fig

def save_all_figures(figure_dict, save_dir='./figures'):
    """保存所有图形到指定目录"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for name, fig in figure_dict.items():
        fig.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{save_dir}/{name}.pdf", bbox_inches='tight')
        print(f"保存图形: {name}")
