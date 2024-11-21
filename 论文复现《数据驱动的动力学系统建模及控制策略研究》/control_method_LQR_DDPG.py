### 2024.11.18
## 基于DDPG算法的逼近最优控制的 DDPG 控制策略

# 基于代码control_method_DDPG.py扩展
#@ 待深度思考的是，如何实现一个通用的单多自由度的训练

import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim
from scipy.linalg import expm, solve_continuous_are, inv, solve_discrete_are

import logging
from datetime import datetime
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
save_time = datetime.now().strftime('%Y_%m_%d')

#from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = ['SimHei', 'Arial']  # 先尝试SimHei显示汉字，再用Arial显示符号等

checkpoint_save_name = "checkpoint.pth" # 检查点文件的保存名称后缀
checkpoint_load_name = "checkpoint.pth" # 希望加载的检查点名称
save_path = './Control_Solution/Control_LQR_DDPG/' # 保存路径设置
model_name = '1119_DDPG_LQR_actormodel' # 保存的模型名称/希望加载的模型名字



## 检查 CUDA 是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## 全局变量
# 参数定义
save_episode = 50 # 多少轮自动保存一次
T = 10 # 系统运行的时间
gamma = 0.98 # 折扣因子
tau = 5e-3 # 网络软更新参数
actor_lr = 5e-4 # 行为策略 actor 网络学习率
critic_lr = 1e-3 # 行为价值 critic 网络学习率
sigma = 0.01 # 高斯噪声标准差
max_episodes = 200 # 最大训练轮次
buffer_size = int(1e6) # 经验池大小
minimal_size = int(5e3) # 训练启动最小值
batch_size = int(64) # 批量学习大小



## 定义一个四阶有阻尼的振动系统的类，基于单自由度改动，原有的尽量不删剪
class VibrationControlEnv:

    ## 系统参数
    def __init__(self, num_dof=1, M = 1, 
                 C = 0.01, 
                 K = 10, 
                 Ts=0.01, tolerance=0.01, T=10):
        self.M = M  # 质量矩阵
        self.C = C  # 阻尼系数
        self.K = K  # 弹簧常数
        self.num_dof = num_dof # 自由度数量
        self.Ts = Ts  # 仿真步长
        self.tolerance = tolerance # 期望的收敛
        self.state = np.zeros(2 * self.num_dof)  #初始状态矩阵 [x1, x2, x3, x4, x1_dot, x2_dot, x3_dot, x4_dot] 
        self.T = T
        self.time = 0.0  # 初始化时间
        
    # 获取LQR系数的K值
    def GetLQR_K(self, Q, R):
        """
        注意为离散值
        计算 LQR 控制增益矩阵 K。
        参数:
            A: 系统状态矩阵 (n x n)
            B: 输入矩阵 (n x m)
            Q: 状态权重矩阵 (n x n)
            R: 输入权重矩阵 (m x m)
        返回:
            K: 反馈增益矩阵 (m x n)
            P: 黎卡提方程的解 (n x n)
        """
        # 求解黎卡提方程
        P = solve_discrete_are(self.A_d, self.B_d, Q, R)
    
        # 计算反馈增益矩阵 K
        self.LQR_K = np.linalg.inv(R + self.B_d.T @ P @ self.B_d) @ (self.B_d.T @ P @ self.A_d)
    def GetStateSpace(self):
        # 建立状态空间
        A_top = np.hstack((np.zeros((self.num_dof, self.num_dof)), np.identity(self.num_dof)))
        A_bottom = np.hstack((-np.linalg.inv(self.M) @ self.K, -np.linalg.inv(self.M) @ self.C))
        self.A = np.vstack((A_top, A_bottom))
        # 创建 B 矩阵
        self.B = np.vstack((np.zeros((self.num_dof, self.num_dof)), -np.linalg.inv(self.M)))
        C = np.hstack((np.identity((self.num_dof)), np.zeros((self.num_dof, self.num_dof))))  # 输出直接是状态变量
        D = np.zeros((self.num_dof, self.num_dof))
        # 状态空间模型
        self.system = StateSpace(A, B, C, D)
        
    def Discretize_System(self):
        """
        离散化连续状态空间系统，包括状态矩阵 A 和输入矩阵 B、E
        参数：
            A: 系统矩阵 (n x n)
            B: 输入矩阵 (n x m)
            E: 动作输入矩阵 (n x p)
            Ts: 采样时间
        返回：
            A_d: 离散状态矩阵
            B_d: 离散输入矩阵
            E_d: 离散动作输入矩阵
        """
        A_top = np.hstack((np.zeros((self.num_dof, self.num_dof)), np.identity(self.num_dof)))
        A_bottom = np.hstack((-np.linalg.inv(self.M) @ self.K, -np.linalg.inv(self.M) @ self.C))
        A = np.vstack((A_top, A_bottom))
        self.A = A
        # 创建 B 矩阵
        B = np.vstack((np.zeros((self.num_dof, self.num_dof)), np.linalg.inv(self.M)))
        B[6,:] = B[7,:] = 0 
        self.B = B
        E = np.vstack((np.zeros((self.num_dof, self.num_dof)), np.linalg.inv(self.M)))
        E[6,:] = E[7,:] = 0
        n = A.shape[0]
        # 扩展复合矩阵
        M = np.block([
            [A, B, E],
            [np.zeros((B.shape[1], A.shape[1] + B.shape[1] + E.shape[1]))],
            [np.zeros((E.shape[1], A.shape[1] + B.shape[1] + E.shape[1]))]
        ])
        # 矩阵指数计算
        M_d = expm(M * self.Ts)
        # 提取 A_d, B_d, E_d
        self.A_d = M_d[:n, :n]
        self.B_d = M_d[:n, n:n + B.shape[1]]
        self.E_d = M_d[:n, n + B.shape[1]:]


    # 环境初始化，可添加初始条件
    def reset(self, x0=None):
        self.Discretize_System()
        #self.GetStateSpace()
        self.state = np.zeros(2 * self.num_dof)
        if x0 is not None:
            self.state = x0 # 初始条件设置
        self.time = 0.0
        u = np.zeros_like(np.arange(0.0, self.T + self.Ts, self.Ts))
        self.u = np.tile(u, (self.num_dof, 1))
        return self.state
    
    # 设置输入
    def u_set(self, u):
        self.u = u

    # 定义每一步接受动作之后更新系统状态
    def step(self, action):
        
        x = self.state
        action = np.hstack((action, [0, 0]))
        
        # 通过当前时间获取应该输入的值
        u = self.u[:,np.where(np.isclose(np.arange(0, self.T+self.Ts, self.Ts), self.time))[0][0]]  # 获取外部激励信号当前时间的值
        
        x = self.A_d @ x.reshape(-1,1).reshape(2*self.num_dof, 1) + self.B_d @ u.reshape(-1,1).reshape(self.num_dof, 1) + self.E_d @ action.reshape(-1,1).reshape(self.num_dof, 1)
        
        self.time += self.Ts
        self.state = x.flatten()
        
        # 定义奖励机制
        x3_squared = x[2]**2
        x4_squared = x[3]**2
        a1_squared = action[0]**2
        a2_squared = action[1]**2

        reward = -10 * (x3_squared + x4_squared) - 0.01 * (a1_squared + a2_squared)
        
        
        done = self.time > self.T+self.Ts*0.3  # 每 T 秒结束一轮仿真 避免时间精度导致的提前结束
        state = self.state[[2,3,6,7]]
        return state, reward, done
    
    # 获取有无控制的响应
    def Controlled_Response(self, u=None, x0=None, actor=None,isLQR=False,Q=None,R=None):
        if x0 is not None:
            state = self.reset(x0=x0)
        if isLQR:
            self.GetLQR_K(Q,R)
        if u is not None:
            self.u_set(u)
        controlled_responses = []
        sum_reward = 0 # 存储奖励
        actions = [] # 存储控制器的输入
        done = False
        while not done:
            # 无控制器则选择输入为零的响应
            if actor: 
                state = torch.tensor(self.state[[2,3,6,7]], dtype=torch.float32).unsqueeze(0).to(device) 
                with torch.no_grad():
                    action = actor(state)
                action = np.clip(action.cpu().numpy().flatten(), -2, 2)
            elif isLQR:
                action = self.LQRControl()
            else:
                action = [0.0, 0.0]
            actions.append(action)
            # 执行没有控制策略的仿真
            _, reward, done = self.step(action)  # 输入为 0，模拟未控制的情况
            controlled_responses.append(self.state)
            sum_reward += reward
        return controlled_responses, actions, sum_reward
    
    # 获取LQR的控制量
    def LQRControl(self):
        cs = self.state[:, np.newaxis]
        action = -self.LQR_K @ self.state[:, np.newaxis]
        return action[[0,1]].flatten()

    
        
## 定义策略网络 Actor
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2) # 输出动作
        self.action_bound = 2 # 输出的控制力范围为(−3N, 3N)

    def forward(self, state):
        x = F.softplus(self.fc1(state))
        action = torch.tanh(self.fc2(x)) * self.action_bound
        return action


## 定义评估价值的网络 Critic
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)  # 输出 Q 值

    def forward(self, state, action):
        x = F.softplus(self.fc1(torch.cat([state, action], dim=-1)))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        q_value = self.fc4(x)
        return q_value


## 定义经验池
class ReplayBuffer:
    def __init__(self, buffer_size=1_000_000, batch_size=64):
        # buffer：用 deque 存储经验数据，每个数据是一个 (state, action, reward, next_state) 元组
        self.buffer = deque(maxlen=buffer_size) 
        self.batch_size = batch_size

    # 将新经验添加到缓冲区
    def store(self, experience):
        self.buffer.append(experience)

    # 采集一个批量数据，准备训练
    def sample(self):
         # 从 buffer 中随机采样一个批次
        batch = random.sample(self.buffer, self.batch_size)
        # 将批次中的各元素分别提取
        states, actions, rewards, next_states = zip(*batch) # 解包得到元组
        
        # 将每个元素转换为张量形式
        states = torch.stack(states).to(device)      # 将 states 元素从 tuple 转换为一个张量
        actions = torch.stack(actions).to(device)    # 将 actions 元素从 tuple 转换为一个张量
        rewards = torch.stack(rewards).unsqueeze(1).to(device)    # 将 rewards 元素从 tuple 转换为一个张量[64,1,1]
        next_states = torch.stack(next_states).to(device)  # 将 next_states 元素从 tuple 转换为一个张量
        
        #print("states shape:", states.shape)
        #print("actions shape:", actions.shape)
        return states, actions, rewards, next_states

    # 返回缓冲区中当前数据的数量
    def __len__(self):
        return len(self.buffer)


## 通用的绘图函数模块
def Plot_Data(figsize=(10, 6), plot_title=None, data_sets=None, x_values=None, colors=None, labels=None, 
              xlabel=None, ylabel=None, xlim=None, ylim=None, log_scale=False, 
              line_styles=None, show_legend=True, legend_loc='upper right', save_path=None, show_grid=False):
    """
    figsize: 绘制的图像大小，默认为(8, 6)\n
    plot_title:图像的标题，默认为空\n
    data_sets: 你要绘制的数据集（可以是多个数组或列表）。若为空则绘图失败\n
    x_values: x轴的值。如果没有提供，默认使用 0, 1, 2, ... 来作为x轴值。\n
    colors: 设置每一条线的颜色，可以是如 ['r', 'g', 'b'] 这样的列表。\n
    labels: 图例的标签。\n
    xlabel=None, ylabel=None :关于横纵轴的标签\n
    xlim, ylim: 设置x轴和y轴的范围，格式为 (min, max)。\n
    log_scale: 如果设置为 True，则y轴使用对数坐标。\n
    line_styles: 每一条线的样式，如 ['-', '--', ':']。\n
    show_legend: 控制是否显示图例。\n
    legend_pos : 指定图例的位置，没有则默认为右上角\n
    save_path: 如果提供该路径，图像将会保存到指定位置，否则会直接显示图像。\n
    show_grid: 是否显示网格，默认为否
    """
    
    # 判断数据集是否正确传入
    if data_sets is None:
        print("未提供数据集，绘图失败")
        return
    
    # 图像的大小
    plt.figure(figsize=figsize)
    
    if plot_title:
        plt.title(plot_title, fontsize = 16)
    
    # 判断是否提供x轴参数，若没有则设置为默认的自然数序列，长度为数据集长度
    if x_values is None:
        x_values = np.arange(len(data_sets[0]))
    
    # 绘制每一个数据集
    for i, data in enumerate(data_sets):
        # 指定绘制数据集的线段属性，没有则默认
        color = colors[i] if colors else None
        label = labels[i] if labels else None
        line_style = line_styles[i] if line_styles else '-'
        plt.plot(x_values, data, color=color, label=label, linestyle=line_style)

    # 开启x和y的标签
    if xlabel: plt.xlabel(xlabel, fontsize = 14)
    if ylabel: plt.ylabel(ylabel, fontsize = 14)
    
    # 确认数据集是否要以对数坐标
    if log_scale:
        plt.yscale('log')  

    # 设置图像的横纵坐标范围
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    # 是否显示图例，默认位置为右上角
    if show_legend and labels:
        plt.legend(loc=legend_loc, fontsize = 12)
    
    # 显示网格选项
    plt.grid(show_grid)

    # 保存路径设置
    if save_path:
        if plot_title:
            plt.savefig(save_path+model_name+'_'+plot_title)
        else:
            plt.savefig(save_path+f'{model_name}_plot')
    else:
        plt.show()


def Target_nnUpdata(actor, critic, target_actor, target_critic):
        # 软更新目标网络参数
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

## DDPG 训练循环函数
def DDPG_circulation(istrain=False, save_path=save_path, model_name=model_name,save_episode=save_episode):
    
    ddpg_sigma = sigma
    
    """
        istrain用来控制是否训练，即执行本函数\n
        sign是标识符，用来确定模型的名称\n
        save_path是模型和训练信息的保存路径，默认为设定的全局参数
    """
    
    if istrain is False:
        return
    
    
    ## 日志定义
    logging.basicConfig(filename=save_path + f'training_log_{current_time}.log',  # 指定日志文件路径
                    level=logging.INFO,          # 设置日志级别
                    format='%(asctime)s - %(levelname)s - %(message)s')  # 设置日志格式
    logging.info(f"Using device: {device}")
    
    
    # 初始化环境、网络、优化器和经验回放池
    env = VibrationControlEnv(num_dof=4, 
                            M = np.diag([1.0, 1.0, 1.0, 1.0]), 
                            C = np.array([[2.0, -1.0, 0.0, 0.0],
                                          [-1.0, 2.0, -1.0, 0.0],
                                          [0.0, -1.0, 2.0, -1.0],
                                          [0.0, 0.0, -1.0, 2.0]]), 
                            K = np.array([[200.0, -100.0, 0.0, 0.0],
                                          [-100.0, 200.0, -100.0, 0.0],
                                          [0.0, -100.0, 200.0, -100.0],
                                          [0.0, 0.0, -100.0, 200.0]]), 
                            Ts=0.01, tolerance=0.01, T=10)
    #env = VibrationControlEnv()
    actor = Actor().to(device)
    critic = Critic().to(device)
    target_actor = Actor().to(device)
    target_critic = Critic().to(device)
    target_actor.load_state_dict(actor.state_dict()) # 将 actor 网络参数复制到目标 actor
    target_critic.load_state_dict(critic.state_dict()) # 将 critic 网络参数复制到目标 critic

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr) # actor 网络的优化器 设置为Adam
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr) # critic 网络的优化器 设置为Adam
    buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size) # 经验回放池
    episode_rewards = [] # 用于记录每一轮的总奖励
    
    
    ## 判断是否需要加载检查点
    start_episode = 0
    # 搜索符合模式的文件
    latest_checkpoint_file = find_latest_checkpoint(save_path)
    load_checkpoint_file = os.path.exists(save_path+checkpoint_load_name)
    if  load_checkpoint_file or latest_checkpoint_file:
        if load_checkpoint_file:
            user_input = input(f"目标的检查点文件 {checkpoint_load_name} 已经找到，是否需要加载？ (y/n): ")
            if user_input.lower() == 'y':
                start_episode, buffer, episode_rewards = load_checkpoint(actor, critic, actor_optimizer, critic_optimizer, buffer, episode_rewards, save_path+checkpoint_load_name)
                Target_nnUpdata(actor, critic, target_actor, target_critic)
            else:
                logging.info("未加载目标的检查点文件，从零开始本次训练")
        if latest_checkpoint_file:
            user_input = input(f"最新的检查点文件 {latest_checkpoint_file} 已经找到，是否需要加载？ (y/n): ")
            if user_input.lower() == 'y':
                start_episode, buffer, episode_rewards = load_checkpoint(actor, critic, actor_optimizer, critic_optimizer, latest_checkpoint_file)
                Target_nnUpdata(actor, critic, target_actor, target_critic)
            else:
                logging.info("未加载最新的检查点文件，从零开始本次训练")
    else:
        print("无检查点文件，从零开始本次训练")
        logging.info("无检查点文件，从零开始本次训练")
    

    # 神经网络训练的循环
    for episode in range(start_episode, max_episodes):
        
        
        # 随训练轮次减少噪声
        if episode+1 >= 0.3 * max_episodes:
            ddpg_sigma = sigma * 0.5 # 减少噪声，提高策略的稳定性
            if episode+1 >= 0.5 * max_episodes:
                ddpg_sigma = 0 # 当轮次进行到一半时，使噪声归为零
        
        
        
        state = env.reset(x0=np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]))# 重置环境并获取初始状态
        state = torch.tensor(state[[2,3,6,7]], dtype=torch.float32).unsqueeze(0).to(device) # 获取第三、四层的状态
        #print("state shape:", state.shape)
        total_reward = 0 # 初始化累积奖励
        done = False # 用于判断回合是否结束
        
        # 设置系统的输入(可变)
        if env.time == 0.0:
            env.state[4] = np.random.choice([-1, 1])
        
        while not done:
    
            with torch.no_grad(): # 禁用梯度计算，加快推理速度
                #print("actor(state) shape:", actor(state).shape)
                action = actor(state) + torch.normal(mean=0, std=ddpg_sigma, size=[1, 1]).to(device) # 获取动作并添加高斯噪声
                #print("action shape:", action.shape)
                
            # np.clip(action.item(), -2, 2) 确保动作在边界内
            next_state, reward, done = env.step(np.clip(action.cpu().numpy().flatten(), -2, 2)) # 执行动作，获取下一个状态、奖励和是否终止
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device) # 转换为张量
            reward = torch.tensor(reward, dtype=torch.float32).to(device) # 转换为张量
            buffer.store((state, action, reward, next_state)) # 将经验存入回放池,注意格式是张量

            if len(buffer) > minimal_size:
                ##  在此更新策略和价值网络...
                
                # 从经验回放池中采样一个批次
                states, actions, rewards, next_states = buffer.sample()
                

                # Critic 网络更新
                # 使用目标 Actor 网络和目标 Critic 网络计算 TD 目标
                with torch.no_grad():
                    target_actions = target_actor(next_states)
                    target_q_values = target_critic(next_states, target_actions)
                    y = rewards + gamma * target_q_values # TD 目标 即奖励 rewards 加上折扣后的未来 Q 值

                # 计算 Critic 网络的预测值
                q_values = critic(states, actions)
                critic_loss = nn.MSELoss()(q_values, y)  # 使用 MSE 计算 Critic 网络损失

                # 反向传播并更新 Critic 网络参数
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor 网络更新
                # 通过 Critic 网络计算 Actor 的损失
                predicted_actions = actor(states)
                actor_loss = -critic(states, predicted_actions).mean()  # 使用策略梯度更新 Actor

                # 反向传播并更新 Actor 网络参数
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 软更新目标网络参数
                Target_nnUpdata(actor, critic, target_actor, target_critic)

                
            total_reward += reward.item() # 累积每个时间步的奖励
            state = next_state # 更新当前状态


        episode_rewards.append(total_reward)



        # 保存检查点
        if (episode+1) % save_episode == 0:  # 每 save_episode 轮保存一次
            save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, buffer, episode_rewards, episode+1)
        
        
        # 打印和绘制每轮的总奖励
        print(f"轮次: {episode+1}, 总奖励: {total_reward}")
        #写入日志
        logging.info(f"轮次: {episode+1}, 总奖励: {total_reward}")


    # 保存训练好的策略模型与总奖励记录集合
    torch.save(actor.state_dict(), save_path + model_name + '.pth')
    np.save(save_path + model_name + '_rewards.npy', np.array(episode_rewards))


## 自动保存训练进度
def save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, buffer, episode_rewards, episode, save_path=save_path, savename=checkpoint_save_name):
    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "buffer":buffer,
        "episode_rewards":episode_rewards,
        "episode": episode
    }
    torch.save(checkpoint, save_path+f'{save_time}_episode{episode}_{savename}') # 保存的名字为 轮次加名字
    logging.info(f"已保存轮次为 {episode} 的检查点文件 '{save_time}_episode{episode}_{savename}'")
    print(f"已保存轮次为 {episode} 的检查点文件 '{save_time}_episode{episode}_{savename}'")
    
## 加载训练状态
def load_checkpoint(actor, critic, actor_optimizer, critic_optimizer, loadfilename, save_path=save_path):
    logging.info(f"正在加载 {loadfilename} 的检查点文件.")
    print(f"正在加载 {loadfilename} 的检查点文件.")
    checkpoint = torch.load(save_path+loadfilename)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    buffer = checkpoint["buffer"]
    episode_rewards = checkpoint["episode_rewards"]
    episode = checkpoint["episode"]
    print(f"加载了轮次为 {episode} 的检查点文件.")
    logging.info(f"加载了轮次为 {episode} 的检查点文件.")
    episode_id = 1
    for episode_reward in episode_rewards:
        logging.info(f"轮次: {episode_id}, 总奖励: {episode_reward} (加载项)")
        episode_id +=1
    logging.info(f"上次训练的奖励数据已写入日志")
    return episode, buffer, episode_rewards


## 查找目录中数字最大的 episode<数字>checkpoint.pth 文件
def find_latest_checkpoint(directory):
    """
    查找目录中数字最大的 episode<数字>checkpoint.pth 文件
    :param directory: 要搜索的目录
    :return: 最大数字对应的文件名，如果没有找到，返回 None
    """
    # 定义正则表达式
    pattern = r"episode(\d+)_checkpoint\.pth"
    latest_file = None
    max_episode = -1

    # 遍历目录中的文件
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            episode = int(match.group(1))  # 提取数字部分
            if episode > max_episode:  # 更新最大数字和文件名
                max_episode = episode
                latest_file = filename

    return latest_file


##生成MLS信号
def Generate_MLS_Signal(time_steps=1000, amplitude=1.0, taps=[0, 1, 3, 4]):
    """
    生成长度为 time_steps 的最大长度序列（MLS）信号。

    参数:
        time_steps (int): 信号长度。
        amplitude (float): 信号幅值。
        taps (list): 反馈多项式的位点，默认为 [0, 1, 3, 4]。

    返回:
        mls_signal (np.ndarray): MLS 信号数组，幅值范围为 [-amplitude, amplitude]。
    """
    # 初始化寄存器，长度为最大 tap 位点 + 1
    state = [1] * (max(taps) + 1)
    mls = []

    for _ in range(time_steps):
        # 根据 taps 计算反馈位
        new_bit = sum([state[tap] for tap in taps]) % 2  # XOR 反馈逻辑
        mls.append(state[-1])  # 保存当前输出
        state = [new_bit] + state[:-1]  # 更新寄存器状态

    # 将信号从 {0, 1} 转换到 [-amplitude, amplitude]
    mls = np.array(mls) * 2 - 1  # 转换为 [-1, 1]
    mls_signal = mls * amplitude  # 调整幅值范围
    return mls_signal


## 主函数
def main():
    
    
        ## DDPG训练循环，若为false则不训练
    DDPG_circulation(istrain=False,save_path=save_path, model_name=model_name)
    
    # 系统的创建
    env = VibrationControlEnv(num_dof=4, 
                            M = np.diag([1.0, 1.0, 1.0, 1.0]), 
                            C = np.array([[2.0, -1.0, 0.0, 0.0],
                                          [-1.0, 2.0, -1.0, 0.0],
                                          [0.0, -1.0, 2.0, -1.0],
                                          [0.0, 0.0, -1.0, 2.0]]), 
                            K = np.array([[200.0, -100.0, 0.0, 0.0],
                                          [-100.0, 200.0, -100.0, 0.0],
                                          [0.0, -100.0, 200.0, -100.0],
                                          [0.0, 0.0, -100.0, 200.0]]), 
                            Ts=0.01, tolerance=0.01, T=10)
    time = np.arange(0.0, env.T, env.Ts)  # 生成 [0, T) 的序列
    time = np.append(time, env.T) # 确保最后包含 T
    

    # 获取LQR控制的仿真
    LQR_responses,LQR_actions,LQR_reward = env.Controlled_Response(x0=np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]), 
                            isLQR=True, Q=np.diag([10,10,0,0,0,0,0,0]), 
                                                  R=np.diag([0.01,0.01,0.01,0.01]))
    LQR_x3 = [LQR_response[2] for LQR_response in LQR_responses] # 第三层位移
    LQR_x4 = [LQR_response[3] for LQR_response in LQR_responses] # 第四层位移
    LQR_v3 = [LQR_response[6] for LQR_response in LQR_responses] # 第三层速度
    LQR_v4 = [LQR_response[7] for LQR_response in LQR_responses] # 第四层速度
    LQR_a1 = [LQR_action[0] for LQR_action in LQR_actions] # 第一层输入
    LQR_a2 = [LQR_action[1] for LQR_action in LQR_actions] # 第二层输入
    
    # 获取无控制的仿真
    uncontrolled_responses,_,_ = env.Controlled_Response(x0=np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]))
    uncontrolled_x3 = [uncontrolled_response[2] for uncontrolled_response in uncontrolled_responses] # 第三层位移
    uncontrolled_x4 = [uncontrolled_response[3] for uncontrolled_response in uncontrolled_responses] # 第四层位移
    uncontrolled_v3 = [uncontrolled_response[6] for uncontrolled_response in uncontrolled_responses] # 第三层速度
    uncontrolled_v4 = [uncontrolled_response[7] for uncontrolled_response in uncontrolled_responses] # 第四层速度
    
    # 获取DDPG控制策略的仿真
        # 加载训练好的策略模型
    actor = Actor().to(device)
    actor.load_state_dict(torch.load(save_path + model_name + '.pth'))
    # 获取有控制的仿真
    DDPG_responses, DDPG_actions, DDPG_reward = env.Controlled_Response(x0=np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]),
                                                                                           actor=actor)
    DDPG_x3 = [DDPG_response[2] for DDPG_response in DDPG_responses] # 第三层位移
    DDPG_x4 = [DDPG_response[3] for DDPG_response in DDPG_responses] # 第四层位移
    DDPG_v3 = [DDPG_response[6] for DDPG_response in DDPG_responses] # 第三层速度
    DDPG_v4 = [DDPG_response[7] for DDPG_response in DDPG_responses] # 第四层速度
    DDPG_a1 = [DDPG_action[0] for DDPG_action in DDPG_actions] # 第一层输入
    DDPG_a2 = [DDPG_action[1] for DDPG_action in DDPG_actions] # 第二层输入
    
    
    ## 绘制训练模型的奖励记录曲线
    # 读取总奖励记录集合
    episode_rewards = np.load(save_path + model_name + '_rewards.npy')
    Plot_Data(
        plot_title='智能体获得的累积回报',
        data_sets=[episode_rewards,[LQR_reward] * episode_rewards.size],
        labels=["DDPG智能体的累计回报","LQR控制的回报"],
        save_path=save_path,
        xlabel="轮次", 
        ylabel='累计奖励',
        xlim=(0,episode_rewards.size),
        legend_loc='lower right'
    )
   
    ## 绘制第n层的位移/速度有无控制对比图
    Plot_Data(
        plot_title='第 3 层位移有无控制对比图',
        x_values=time,
        data_sets=[uncontrolled_x3, LQR_x3, DDPG_x3],
        labels=["无控制", "LQR控制", "DDPG控制"],
        save_path=save_path,
        xlabel="时间(s)", 
        ylabel='位移',
        xlim=(0,env.T)
    )
    Plot_Data(
        plot_title='第 4 层位移有无控制对比图',
        x_values=time,
        data_sets=[uncontrolled_x4, LQR_x4, DDPG_x4],
        labels=["无控制", "LQR控制", "DDPG控制"],
        save_path=save_path,
        xlabel="时间(s)", 
        ylabel='位移',
        xlim=(0,env.T)
    )
    Plot_Data(
        plot_title='第 3 层速度有无控制对比图',
        x_values=time,
        data_sets=[uncontrolled_v3, LQR_v3, DDPG_v3],
        labels=["无控制", "LQR控制", "DDPG控制"],
        save_path=save_path,
        xlabel="时间(s)", 
        ylabel='速度',
        xlim=(0,env.T)
    )
    Plot_Data(
        plot_title='第 4 层速度有无控制对比图',
        x_values=time,
        data_sets=[uncontrolled_v4, LQR_v4, DDPG_v4],
        labels=["无控制", "LQR控制", "DDPG控制"],
        save_path=save_path,
        xlabel="时间(s)", 
        ylabel='速度',
        xlim=(0,env.T)
    )
    
    ## 绘制控制输出
    Plot_Data(
        plot_title='作用于第 1 层的输入',
        x_values=time,
        data_sets=[LQR_a1, DDPG_a1],
        labels=["LQR控制", "DDPG控制"],
        save_path=save_path,
        xlabel="时间(s)", 
        ylabel='输入力',
        xlim=(0,env.T)
    )
    ## 绘制控制输出
    Plot_Data(
        plot_title='作用于第 2 层的输入',
        x_values=time,
        data_sets=[LQR_a2, DDPG_a2],
        labels=["LQR控制", "DDPG控制"],
        save_path=save_path,
        xlabel="时间(s)", 
        ylabel='输入力',
        xlim=(0,env.T)
    )
    
   

    ## MLS工况仿真
    # mls_signal = Generate_MLS_Signal(time_steps=time.size, amplitude=2.5)
    # uncontrolled_responses_mls,_ = env.Controlled_Response(mls_signal)
    # controlled_responses_mls,actions_mls = env.Controlled_Response(mls_signal, actor)
    
    # # 绘制有无控制策略的系统响应仿真的图
    # Plot_Data(
    #     plot_title='线性弹簧振子控制前后的位移(MLS信号)',
    #     xlim=(0,env.T),
    #     labels=['无控制MLS响应','控制后响应'],
    #     x_values=time,
    #     data_sets=[uncontrolled_responses_mls, controlled_responses_mls],
    #     save_path=save_path
    #     )
    
    # # 绘制外激力与控制力的输入曲线
    # Plot_Data(
    #     plot_title='外激力输入与控制力响应(MLS信号)',
    #     xlim=(0,env.T),
    #     labels=['MLS外激力','控制力'],
    #     x_values=time,
    #     data_sets=[mls_signal, actions_mls],
    #     line_styles=[None, '--'],
    #     save_path=save_path
    #     )
    
    plt.show()


if __name__ == "__main__":
    main()