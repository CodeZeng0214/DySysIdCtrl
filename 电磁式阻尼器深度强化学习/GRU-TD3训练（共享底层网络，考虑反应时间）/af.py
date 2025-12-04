# 辅助函数定义
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import logging
from TD3 import TD3Agent, Gru_TD3Agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 枚举观测状态
STATES_NAME = {
    0: "吸振器位移 (x1)",
    1: "吸振器速度 (v1)",
    2: "吸振器加速度 (a1)",
    3: "平台位移 (x2)",
    4: "平台速度 (v2)",
    5: "平台加速度 (a2)"
}

class Datasets:
    """管理数据的类"""
    def __init__(self):
        self.checkpoint_name = "空检查点"

        # 单轮次数据
        self.state_history = np.zeros((0, 6))  # 单轮次的状态历史
        self.action_history = np.array([])  # 单轮次的动作历史
        self.reward_history = np.array([])  # 单轮次的奖励历史
        self.dt_history = np.array([])  # 单轮次的时间步长历史
        self.time_history = np.array([])  # 单轮次的时间历史
        self.delay_time = np.array([])  # 单轮次的时延历史

        # 所有轮次数据
        self.current_episode = 0
        self.episode_rewards = np.array([])  # 轮次的累计奖励
        self.episode_simu_rewards = np.array([])  # 轮次的仿真累计奖励
        self.episode_actor_losses = np.array([])  # 轮次的策略网络累计损失
        self.episode_critic_losses = np.array([])  # 轮次的评估网络累计损失
        self.episode_epsilons = np.array([])  # 轮次的探索率历史

    def save_datasets(self, agent: Union[TD3Agent, Gru_TD3Agent], save_dir):
        """保存数据集"""
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "checkpoint_name": self.checkpoint_name,

            # 保存数据
            "state_history": self.state_history,
            "action_history": self.action_history,
            "reward_history": self.reward_history,
            "dt_history": self.dt_history,
            "time_history": self.time_history,
            "delay_time": self.delay_time,
            
            "current_episode": self.current_episode,
            "episode_rewards": self.episode_rewards,
            "episode_simu_rewards": self.episode_simu_rewards,
            "episode_epsilons": self.episode_epsilons,
            "episode_actor_losses": self.episode_actor_losses,
            "episode_critic_losses": self.episode_critic_losses,
            # # 保存模型参数
            # 'agent': agent,
            'modelnn': agent.modelnn,
            # 'gru_predictor': agent.gru_predictor.state_dict() if isinstance(agent, Gru_TD3Agent) else None,
            # 'gru_predictor2': agent.gru_predictor2.state_dict() if isinstance(agent, Gru_TD3Agent) else None,
            # 'target_gru_predictor': agent.target_gru_predictor.state_dict() if isinstance(agent, Gru_TD3Agent) else None,
            # 'target_gru_predictor2': agent.target_gru_predictor2.state_dict() if isinstance(agent, Gru_TD3Agent) else None,
            # 'actor': agent.actor.state_dict(),
            # 'critic1': agent.critic1.state_dict(),
            # 'critic2': agent.critic2.state_dict(),
            # 'target_actor': agent.target_actor.state_dict(),
            # 'target_critic1': agent.target_critic1.state_dict(),
            # 'target_critic2': agent.target_critic2.state_dict(),
            # 'actor_optimizer': agent.actor_optimizer.state_dict(),
            # 'critic1_optimizer': agent.critic1_optimizer.state_dict(),
            # 'critic2_optimizer': agent.critic2_optimizer.state_dict(),
            # 'total_it': agent.total_it
        }, os.path.join(save_dir, f"{self.checkpoint_name}.pth"))
        logging.info(f"保存检查点: {os.path.join(save_dir, f'{self.checkpoint_name}.pth')}")

    def load_datasets(self, save_dir, agent:Union[TD3Agent, Gru_TD3Agent])-> int:
        """加载训练数据和模型参数"""
        checkpoint_files = find_checkpoint_files(save_dir)
        load_previous_model = False
        if checkpoint_files: 
            load_previous_model = input("是否加载先前的训练模型? (y/n): ").strip().lower() == 'y' or ''
        else:
            return 0

        if load_previous_model:
            logging.info(f"准备加载检查点文件")
            
            if checkpoint_files:
                print("\n找到以下检查点文件:")
                for i, file in enumerate(checkpoint_files):
                    file_name = os.path.basename(file)
                    print(f"{i+1}. {file_name}")
                
                choice = input("请选择要加载的检查点文件编号 (输入数字，直接回车取最新): ")
                
                if choice.strip():
                    try:
                        selected_index = int(choice) - 1
                        if 0 <= selected_index < len(checkpoint_files):
                            checkpoint_path = checkpoint_files[selected_index]
                        else:
                            print("无效的选择，使用最新检查点")
                            checkpoint_path = checkpoint_files[0]
                    except ValueError:
                        print("无效输入，使用最新检查点")
                        checkpoint_path = checkpoint_files[0]
                else:
                    checkpoint_path = checkpoint_files[0]

                self.checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

                print(f"加载检查点: {self.checkpoint_name}")
                logging.info(f"加载检查点: {self.checkpoint_name}")

                # 加载模型并获取训练状态
                datasets = torch.load(os.path.join(save_dir, f"{self.checkpoint_name}.pth"), map_location=device, weights_only=False)
                print(f"✅ 检查点文件加载成功")
                logging.info(f"✅ 检查点文件加载成功")

                # 加载训练数据
                self.current_episode = datasets.get("current_episode", 0)
                self.state_history = datasets.get("state_history", np.zeros((0, 6)))
                self.action_history = datasets.get("action_history", np.array([]))
                self.reward_history = datasets.get("reward_history", np.array([]))
                self.dt_history = datasets.get("dt_history", np.array([]))
                self.time_history = datasets.get("time_history", np.array([]))
                self.delay_time = datasets.get("delay_time", np.array([]))
                self.episode_rewards = datasets.get("episode_rewards", np.array([]))
                self.episode_simu_rewards = datasets.get("episode_simu_rewards", np.array([]))
                self.episode_epsilons = datasets.get("episode_epsilons", np.array([]))
                self.episode_actor_losses = datasets.get("episode_actor_losses", np.array([]))
                self.episode_critic_losses = datasets.get("episode_critic_losses", np.array([]))

                # # 加载模型参数
                try:
                    # load_agent: Union[None, TD3Agent, Gru_TD3Agent] = datasets["agent"] # 废掉的方法，此方法会导致agent的参数被覆盖
                    modelnn: List[torch.nn.Module] = datasets["modelnn"]
                    for model, state_dict in zip(agent.modelnn, modelnn):
                        model.load_state_dict(state_dict.state_dict())
                #     agent.gru_predictor.load_state_dict(datasets["gru_predictor"]) if isinstance(agent, Gru_TD3Agent) and datasets["gru_predictor"] is not None else None
                #     agent.gru_predictor2.load_state_dict(datasets["gru_predictor2"]) if isinstance(agent, Gru_TD3Agent) and datasets["gru_predictor2"] is not None else None
                #     agent.target_gru_predictor.load_state_dict(datasets["target_gru_predictor"]) if isinstance(agent, Gru_TD3Agent) and datasets["target_gru_predictor"] is not None else None
                #     agent.target_gru_predictor2.load_state_dict(datasets["target_gru_predictor2"]) if isinstance(agent, Gru_TD3Agent) and datasets["target_gru_predictor2"] is not None else None
                #     agent.actor.load_state_dict(datasets["actor"])
                #     agent.critic1.load_state_dict(datasets["critic1"])
                #     agent.critic2.load_state_dict(datasets["critic2"])
                #     agent.target_actor.load_state_dict(datasets["target_actor"])
                #     agent.target_critic1.load_state_dict(datasets["target_critic1"])
                #     agent.target_critic2.load_state_dict(datasets["target_critic2"])
                #     agent.actor_optimizer.load_state_dict(datasets["actor_optimizer"])
                #     agent.critic1_optimizer.load_state_dict(datasets["critic1_optimizer"])
                #     agent.critic2_optimizer.load_state_dict(datasets["critic2_optimizer"])
                #     agent.total_it = datasets["total_it"]
                except Exception as e:
                    print(f"加载模型参数时发生错误: {e}")
                    logging.error(f"加载模型参数时发生错误: {e}")
                print(f"成功加载检查点: {self.checkpoint_name}，当前回合: {self.current_episode}")
                logging.info(f"成功加载检查点: {self.checkpoint_name}, 当前回合: {self.current_episode}")
            else:
                print("未找到可加载的检查点文件")
                logging.info("未找到可加载的检查点")
        return self.current_episode

    def record_history(self, state: np.ndarray, action: float, reward: float, dt: float, time: float, delay_time: float=0):
        """记录单个回合的当前时间步数据"""
        self.state_history = np.vstack([self.state_history, state])
        self.action_history = np.append(self.action_history, action)
        self.reward_history = np.append(self.reward_history, reward)
        self.dt_history = np.append(self.dt_history, dt)
        self.time_history = np.append(self.time_history, time)
        self.delay_time = np.append(self.delay_time, delay_time)

    def reset_episode_data(self):
        """重置单回合历史数据"""
        self.state_history = np.zeros((0, 6))  # 单轮次的状态历史
        self.reward_history = np.array([])
        self.action_history = np.array([])
        self.dt_history = np.array([])
        self.time_history = np.array([])
        
    def reset_episode_history(self):
        """重置轮次历史，保留模型参数"""
        self.reset_episode_data()
        self.current_episode = 0
        self.episode_rewards = np.array([])  # 轮次的累计奖励
        self.episode_simu_rewards = np.array([])  # 轮次的仿真累计奖励
        self.episode_epsilons = np.array([])  # 轮次的预测器累计损失
        self.episode_actor_losses = np.array([])  # 轮次的策略网络累计损失
        self.episode_critic_losses = np.array([])  # 轮次的评估网络累计损失

    def record_episode_data(self, episode_reward, episode_sim_reward, episode_actor_losses, episode_critic_losses, epsilon):
        """记录回合累计数据"""
        self.episode_rewards = np.append(self.episode_rewards, episode_reward)
        self.episode_simu_rewards = np.append(self.episode_simu_rewards, episode_sim_reward)
        self.episode_actor_losses = np.append(self.episode_actor_losses, episode_actor_losses)
        self.episode_critic_losses = np.append(self.episode_critic_losses, episode_critic_losses)
        self.episode_epsilons = np.append(self.episode_epsilons, epsilon)

    def plot_episode_history(self, plot_state=[], plot_action=False, plot_reward=False, plot_dt=False, plot_delay_time=False, save_path=None, show=False):
        """绘制训练历史\n
        - plot_state: 是否绘制状态历史，[状态索引]
        - plot_action: 是否绘制动作历史
        - plot_reward: 是否绘制奖励历史
        - plot_dt: 是否绘制时间步长历史
        """
        if plot_state and self.state_history.size > 0:
            for state_idx in plot_state:
                plot_data(x_values_list=self.time_history, y_values_list=self.state_history[:, state_idx], 
                          plot_title=f'{self.checkpoint_name} 状态 {STATES_NAME[state_idx]} 历史', legends=[f'状态 {STATES_NAME[state_idx]}'], 
                          xlabel='时间 (s)', ylabel=f'状态 {STATES_NAME[state_idx]}', save_path=save_path, show=show)

        if plot_action and self.action_history.size > 0:
            plot_data(x_values_list=self.time_history, y_values_list=self.action_history, 
                      plot_title=f'{self.checkpoint_name} 动作历史', legends=['动作'], 
                      xlabel='时间 (s)', ylabel='动作', save_path=save_path, show=show)

        if plot_reward and self.reward_history.size > 0:
            plot_data(x_values_list=self.time_history, y_values_list=self.reward_history, 
                      plot_title=f'{self.checkpoint_name} 奖励历史', legends=['奖励'], 
                      xlabel='时间 (s)', ylabel='奖励', save_path=save_path, show=show)

        if plot_dt and self.dt_history.size > 0:
            plot_data(x_values_list=self.time_history, y_values_list=self.dt_history, 
                      plot_title=f'{self.checkpoint_name} 时间步长历史', legends=['时间步长'], 
                      xlabel='时间 (s)', ylabel='时间步长', save_path=save_path, show=show)
            
        if plot_delay_time and self.delay_time.size > 0:
            plot_data(x_values_list=self.time_history, y_values_list=self.delay_time, 
                      plot_title=f'{self.checkpoint_name} 时延历史', legends=['时延'], 
                      xlabel='时间 (s)', ylabel='时延', save_path=save_path, show=show)

    def plot_episode_datas(self, plot_rewards=True, plot_simu_rewards=False, plot_predictor_losses=False, plot_actor_losses=False, plot_critic_losses=False, save_path=None, show=False):
        """绘制回合数据"""
        if plot_rewards and self.episode_rewards.size > 0:
            plot_data(x_values_list=np.arange(len(self.episode_rewards)), y_values_list=self.episode_rewards, 
                      plot_title=f'{self.checkpoint_name} 奖励历史', legends=['奖励'], 
                      xlabel='回合', ylabel='奖励', save_path=save_path, show=show)
            
        if plot_simu_rewards and self.episode_simu_rewards.size > 0:
            plot_data(x_values_list=np.arange(len(self.episode_simu_rewards)), y_values_list=self.episode_simu_rewards, 
                      plot_title=f'{self.checkpoint_name} 仿真奖励历史', legends=['仿真奖励'], 
                      xlabel='回合', ylabel='仿真奖励', save_path=save_path, show=show)
            
        if plot_predictor_losses and self.episode_epsilons.size > 0:
            plot_data(x_values_list=np.arange(start=np.count_nonzero(self.episode_epsilons == 0), stop=len(self.episode_epsilons)), y_values_list=self.episode_epsilons, 
                      plot_title=f'{self.checkpoint_name} 预测器损失历史', legends=['预测器损失'], 
                      xlabel='回合', ylabel='损失', save_path=save_path, show=show)
            
        if plot_actor_losses and self.episode_actor_losses.size > 0:
            plot_data(x_values_list=np.arange(start=np.count_nonzero(self.episode_actor_losses == 0), stop=len(self.episode_actor_losses)), y_values_list=self.episode_actor_losses[self.episode_actor_losses != 0], 
                      plot_title=f'{self.checkpoint_name} Actor损失历史', legends=['Actor损失'], 
                      xlabel='回合', ylabel='损失', save_path=save_path, show=show)

        if plot_critic_losses and self.episode_critic_losses.size > 0:
            plot_data(x_values_list=np.arange(start=np.count_nonzero(self.episode_critic_losses == 0), stop=len(self.episode_critic_losses)), y_values_list=self.episode_critic_losses[self.episode_critic_losses != 0], 
                      plot_title=f'{self.checkpoint_name} Critic损失历史', legends=['Critic损失'], 
                      xlabel='回合', ylabel='损失', save_path=save_path, show=show)

def plot_compare_no_control(nc_datasets:Datasets, c_datasets:Datasets, plot_state=False, save_path=None, plot_dt=False, plot_delay_time=False, show=False):
    """绘制与无控制的状态比较图"""
    if plot_state and nc_datasets.state_history.size > 0 and c_datasets.state_history.size > 0:
        for state_idx in plot_state:
            plot_data(x_values_list=[nc_datasets.time_history, c_datasets.time_history],
                    y_values_list=[nc_datasets.state_history[:, state_idx], c_datasets.state_history[:, state_idx]], 
                    plot_title=f'{c_datasets.checkpoint_name} 状态 {STATES_NAME[state_idx]} 对比', legends=['无控制', 'TD3控制'], 
                    xlabel='时间 (s)', ylabel=f'状态 {STATES_NAME[state_idx]}', save_path=save_path, show=show)

    c_datasets.plot_episode_history(plot_state=False, plot_action=True, plot_reward=True, plot_dt=plot_dt, plot_delay_time=plot_delay_time, save_path=save_path, show=show)

def plot_state_comparison(results_no_control, results_td3, save_path=None):
    """比较不同控制策略下的状态轨迹
    
    Args:
        results_no_control: 无控制的仿真结果
        results_td3: TD3控制的仿真结果
        save_path: 图表保存路径
    
    Returns:
        fig: 图表对象
    """
    # 绘制位移和速度对比图
    fig = plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 0], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 0], label='LQR')
    plt.plot(results_td3['times'], results_td3['states'][:, 0], label='TD3')
    plt.xlabel('Time (s)')
    plt.ylabel('Position x1')
    plt.title('吸振器位移')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 2], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 2], label='LQR')
    plt.plot(results_td3['times'], results_td3['states'][:, 2], label='TD3')
    plt.xlabel('Time (s)')
    plt.ylabel('Position x2')
    plt.title('平台位移')
    plt.legend()
    plt.grid(True)
    
    # 绘制速度
    plt.subplot(2, 2, 3)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 1], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 1], label='LQR')
    plt.plot(results_td3['times'], results_td3['states'][:, 1], label='TD3')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity v1')
    plt.title('吸振器速度')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 3], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 3], label='LQR')
    plt.plot(results_td3['times'], results_td3['states'][:, 3], label='TD3')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity v2')
    plt.title('平台速度')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    # 绘制控制输入
    fig2 = plt.figure(figsize=(10, 5))
    # plt.plot(results_lqr['times'][:-1], results_lqr['actions'], label='LQR Control')
    plt.plot(results_td3['times'][:-1], results_td3['actions'], label='TD3 Control')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input')
    plt.title('控制输入对比')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        # 保存控制输入图
        control_save_path = save_path.replace('.png', '_control.png')
        plt.savefig(control_save_path)
        
    plt.show()
    
    return fig
    
def find_checkpoint_files(directory):
    """查找目录中所有检查点文件"""
    checkpoint_files = []
    for filename in os.listdir(directory):
        if filename.endswith('_checkpoint.pth'):
            checkpoint_files.append(os.path.join(directory, filename))
    
    # 按文件修改时间排序
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoint_files

## 通用的绘图函数模块
def plot_data(x_values_list: List[np.ndarray], y_values_list: List[np.ndarray], figsize=(10, 6), plot_title: str = None, 
              colors: List[str] = None, line_styles: List[str] = None, 
              legends: List[str] = None, show_legend=True, legend_loc='upper right', 
              xlabel: str = None, ylabel: str = None, log_scale=False, 
              xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None, 
              save_path: str = None, show_grid=False, show=True):
    """
    - x_values: 绘制的x轴数据，可以是一个数组或多个数组列表。
    - y_values: 绘制的y轴数据，可以是一个数组或多个数组列表。
    - figsize: 图像的大小, 默认为(10, 6)
    - plot_title: 图像的标题, 默认为None
    - colors: 设置每一条线的颜色，可以是如 ['r', 'g', 'b'] 这样的列表。
    - line_styles: 每一条线的样式，如 ['-', '--', ':']。
    - legends: 图例的标签。
    - show_legend: 控制是否显示图例。
    - legend_loc : 指定图例的位置，没有则默认为右上角
    - xlabel=None, ylabel=None :关于横纵轴的标签
    - log_scale: 如果设置为 True，则y轴使用对数坐标。
    - xlim, ylim: 设置x轴和y轴的范围，格式为 (min, max)。
    - save_path: 如果提供该路径，图像将会保存到指定位置，否则会直接显示图像。
    - show_grid: 是否显示网格，默认为否
    - show: 是否显示图像，默认为是
    """
    # 图像的大小
    plt.figure(figsize=figsize)

    # 设置标题
    if plot_title:
        plt.title(plot_title, fontsize = 16)

    # 判断是否提供x轴参数，若没有则设置为默认的自然数序列，长度为数据集长度
    if x_values_list is None:
        x_values_list = [np.arange(len(y_values_list[i])) for i in range(len(y_values_list))]

    # 确保x_values_list 和 y_values_list 是一个列表
    if not isinstance(x_values_list, list): x_values_list = [x_values_list]
    if not isinstance(y_values_list, list): y_values_list = [y_values_list]
        
    # 绘制每一个数据集
    for i, data in enumerate(y_values_list):
        # 获取x轴数据
        if len(x_values_list) == 1: x_values = x_values_list[0]
        else: x_values = x_values_list[i]

        # 检查x轴数据与y轴数据的长度是否匹配
        if len(x_values) != len(data):
            print("x_values的长度与数据集长度不匹配，绘图失败")
            return
        
        # 指定绘制数据集的线段属性，没有则默认
        color = colors[i] if colors else None
        label = legends[i] if legends else None
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
    if show_legend and legends:
        plt.legend(loc=legend_loc, fontsize = 12)
    
    # 显示网格选项
    plt.grid(show_grid)

    # 保存路径设置
    if save_path:
        if plot_title: plt.savefig(save_path +'\\'+ f"{plot_title}.png")
        else: plt.savefig(save_path +'\\'+ 'plot.png')
    
    if show:
        plt.show()
    else:
        plt.close()