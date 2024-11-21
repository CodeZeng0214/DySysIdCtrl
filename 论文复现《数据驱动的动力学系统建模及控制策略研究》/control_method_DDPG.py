### 2024.11.15-11.17
## 基于DDPG算法的控制策略算法
## 此代码实现的是论文 蒋纪元《数据驱动的动力学系统建模及控制策略研究》的 5.2 节基于 DDPG 的振动控制算法

#@ （已经实现）神经网络训练的中断与延续
#@ （已经实现）思考一个通用的绘图函数模块
#@ （已解决）action的限制问题，在张量之前还是之后？
#@ 存在的问题，加载轮次靠后的模型由于模型过大会导致加载时长过长，如何简化加载的数据？

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
import logging
from datetime import datetime
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
save_time = datetime.now().strftime('%Y_%m_%d')

#from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = ['SimHei', 'Arial']  # 先尝试SimHei显示汉字，再用Arial显示符号等

checkpoint_save_name = "checkpoint.pth" # 检查点文件的保存名称后缀
checkpoint_load_name = "checkpoint.pth" # 希望加载的检查点名称
save_path = './Control_Solution/Control_DDPG/' # 保存路径设置
model_name = '1118_DDPG_actormodel' # 保存的模型名称/希望加载的模型名字



## 检查 CUDA 是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## 全局变量
# 参数定义
save_episode = 20 # 多少轮自动保存一次
T = 10 # 系统运行的时间
gamma = 0.98 # 折扣因子
tau = 5e-3 # 网络软更新参数
actor_lr = 5e-4 # 行为策略 actor 网络学习率
critic_lr = 1e-3 # 行为价值 critic 网络学习率
sigma = 0.01 # 高斯噪声标准差
max_episodes = 500 # 最大训练轮次
buffer_size = int(1e6) # 经验池大小
minimal_size = int(5e3) # 训练启动最小值
batch_size = int(64) # 批量学习大小



## 定义一个二阶有阻尼的振动系统的类
class VibrationControlEnv:

    ## 系统参数
    def __init__(self, m=1, c=0.001, k=10, Ts=0.01, tolerance=0.01, T=10):
        self.m = m  # 质量
        self.c = c  # 阻尼系数
        self.k = k  # 弹簧常数
        self.Ts = Ts  # 仿真步长
        self.tolerance = tolerance # 期望的收敛
        self.state = np.array([0.0, 0.0])  # 初始状态 [x, x_dot]
        self.T = T
        self.time = 0.0  # 初始化时间
        self.u = np.zeros_like(np.arange(0.0, self.T + self.Ts, self.Ts)) # 初始化输入

    # 环境初始化
    def reset(self):
        self.state = np.array([0.0, 0.0])
        self.time = 0.0
        self.u = np.zeros_like(np.arange(0.0, self.T + self.Ts, self.Ts)) # 初始化输入
        return self.state
    
    # 设置输入
    def u_set(self, u):
        self.u = u

    # 定义每一步接受动作之后更新系统状态
    def step(self, action):
        x, x_dot = self.state
        
        # 通过当前时间获取应该输入的值
        u = self.u[np.where(np.isclose(np.arange(0, self.T+self.Ts, self.Ts), self.time))[0][0]]  # 获取外部激励信号当前时间的值
        
        x_t = x # 存储当前时刻的位置
        
        # 二阶微分方程离散化
        x_ddot = (u - self.c * x_dot - self.k * x + action) / self.m
        x_dot = x_dot + x_ddot * self.Ts
        x = x + x_dot * self.Ts
        x_t1 = x # 存储更新系统后的位置
        self.state = np.array([x, x_dot])
        self.time += self.Ts

        # 定义奖励机制
        if abs(x_t1) < self.tolerance and abs(x_t1) < abs(x_t):
            reward = abs(self.tolerance - abs(x_t1) / self.tolerance)
        else:
            if abs(x_t1) > self.tolerance and abs(x_t1) > abs(x_t):
                reward = -0.5 
            else:
                reward = 0
        done = self.time > self.T+self.Ts*0.3  # 每 T 秒结束一轮仿真 避免时间精度导致的提前结束
        return self.state, reward, done
    
    # 获取有无控制的响应
    def Controlled_Response(self, u, actor=None):
        
        state = self.reset()
        self.u_set(u)
        controlled_responses = []
        actions = [] # 存储控制器的输入
        done = False
        while not done:
            # 无控制器则选择输入为零的响应
            if actor: 
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) 
                action = actor(state)
                action = action.item()
                action = np.clip(action, -3, 3)
                actions.append(action)
            else:
                action = 0.0
            # 执行没有控制策略的仿真
            state, _, done = self.step(action)  # 输入为 0，模拟未控制的情况
            controlled_responses.append(state[0])
        return controlled_responses, actions
    
        
## 定义策略网络 Actor
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 1) # 输出动作
        self.action_bound = 3 # 输出的控制力范围为(−3N, 3N)

    def forward(self, state):
        x = F.softplus(self.fc1(state))
        action = torch.tanh(self.fc2(x)) * self.action_bound
        return action


## 定义评估价值的网络 Critic
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(3, 64)
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
    env = VibrationControlEnv()
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
        
        
        
        state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).to(device) # 重置环境并获取初始状态
        #print("state shape:", state.shape)
        total_reward = 0 # 初始化累积奖励
        done = False # 用于判断回合是否结束
        
        # 设置系统的输入
        u = np.sin(2 * np.pi * np.arange(0, env.T+env.Ts, env.Ts))
        env.u_set(u)
        
        while not done:
    
            with torch.no_grad(): # 禁用梯度计算，加快推理速度
                #print("actor(state) shape:", actor(state).shape)
                action = actor(state) + torch.normal(mean=0, std=ddpg_sigma, size=[1, 1]).to(device) # 获取动作并添加高斯噪声
                #print("action shape:", action.shape)
                
            # np.clip(action.item(), -3, 3) 确保动作在边界内
            next_state, reward, done = env.step(np.clip(action.item(), -3, 3)) # 执行动作，获取下一个状态、奖励和是否终止
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device) # 转换为张量
            reward = torch.tensor([reward], dtype=torch.float32).to(device) # 转换为张量
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
    
    
    ## 绘制训练模型的奖励记录曲线
    # 读取总奖励记录集合
    episode_rewards = np.load(save_path + model_name + '_rewards.npy')
    Plot_Data(
        plot_title='DDPG 算法累积回报',
        data_sets=[episode_rewards],
        save_path=save_path,
        xlabel="轮次", 
        ylabel='累计奖励',
    )
   
    
    ## 绘制有无控制策略的系统响应仿真
    
    # 获取无控制的仿真
    env = VibrationControlEnv(T=20)
    time = np.arange(0.0, env.T, env.Ts)  # 生成 [0, T) 的序列
    time = np.append(time, env.T) # 确保最后包含 T
    # np.arange()函数的语法是np.arange(start, stop, step)，它会生成一个从start（包含）开始，到stop（不包含）结束，以step为间隔的序列。
    u = np.sin(2 * np.pi * time)
    uncontrolled_responses,_ = env.Controlled_Response(u)
    
   
    # 加载训练好的策略模型
    actor = Actor().to(device)
    actor.load_state_dict(torch.load(save_path + model_name + '.pth'))
    # 获取有控制的仿真
    controlled_responses, actions = env.Controlled_Response(u, actor=actor)
    
    # 绘制有无控制策略的系统响应仿真的图
    Plot_Data(
        plot_title='线性弹簧振子控制前后的位移(训练信号)',
        xlim=(0,env.T),
        labels=['无控制响应','控制后响应'],
        x_values=time,
        data_sets=[uncontrolled_responses, controlled_responses],
        save_path=save_path
        )
    
    # 绘制外激力与控制力的输入曲线
    Plot_Data(
        plot_title='外激力输入与控制力响应(训练信号)',
        xlim=(0,env.T),
        labels=['外激力','控制力'],
        x_values=time,
        data_sets=[u, actions],
        line_styles=[None, '--'],
        save_path=save_path
        )
    
    ## 获取MLS信号的输入

    mls_signal = Generate_MLS_Signal(time_steps=time.size, amplitude=2.5)
    uncontrolled_responses_mls,_ = env.Controlled_Response(mls_signal)
    controlled_responses_mls,actions_mls = env.Controlled_Response(mls_signal, actor)
    
    # 绘制有无控制策略的系统响应仿真的图
    Plot_Data(
        plot_title='线性弹簧振子控制前后的位移(MLS信号)',
        xlim=(0,env.T),
        labels=['无控制MLS响应','控制后响应'],
        x_values=time,
        data_sets=[uncontrolled_responses_mls, controlled_responses_mls],
        save_path=save_path
        )
    
    # 绘制外激力与控制力的输入曲线
    Plot_Data(
        plot_title='外激力输入与控制力响应(MLS信号)',
        xlim=(0,env.T),
        labels=['MLS外激力','控制力'],
        x_values=time,
        data_sets=[mls_signal, actions_mls],
        line_styles=[None, '--'],
        save_path=save_path
        )
    
    plt.show()


if __name__ == "__main__":
    main()