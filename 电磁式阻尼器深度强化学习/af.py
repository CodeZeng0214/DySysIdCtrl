# 辅助函数定义

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import logging
from ddpg_agent import DDPGAgent, GruDDPGAgent

def plot_rewards(rewards, avg_rewards=None, window=10, save_dir=None, save_path=None):
    """绘制奖励曲线
    
    Args:
        rewards: 每轮的奖励列表
        avg_rewards: 平均奖励列表，如果为None则计算移动平均
        window: 计算移动平均的窗口大小
        save_dir: 保存目录（旧参数，为向后兼容保留）
        save_path: 完整保存路径，优先于save_dir
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward', alpha=0.5)
    
    if avg_rewards is None and len(rewards) > window:
        # 计算移动平均
        avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(rewards)), avg_rewards, label=f'{window}-Episode Average')
    elif avg_rewards is not None:
        plt.plot(avg_rewards, label='Average Reward')
        
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path)
    elif save_dir:
        plt.savefig(os.path.join(save_dir, 'training_rewards.png'))
        
    plt.show()
    
    return plt.gcf()

    
def plot_state_comparison(results_no_control, results_ddpg, save_path=None):
    """比较不同控制策略下的状态轨迹
    
    Args:
        results_no_control: 无控制的仿真结果
        results_ddpg: DDPG控制的仿真结果
        save_path: 图表保存路径
    
    Returns:
        fig: 图表对象
    """
    # 绘制位移和速度对比图
    fig = plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 0], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 0], label='LQR')
    plt.plot(results_ddpg['times'], results_ddpg['states'][:, 0], label='DDPG')
    plt.xlabel('Time (s)')
    plt.ylabel('Position x1')
    plt.title('吸振器位移')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 2], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 2], label='LQR')
    plt.plot(results_ddpg['times'], results_ddpg['states'][:, 2], label='DDPG')
    plt.xlabel('Time (s)')
    plt.ylabel('Position x2')
    plt.title('平台位移')
    plt.legend()
    plt.grid(True)
    
    # 绘制速度
    plt.subplot(2, 2, 3)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 1], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 1], label='LQR')
    plt.plot(results_ddpg['times'], results_ddpg['states'][:, 1], label='DDPG')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity v1')
    plt.title('吸振器速度')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 3], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 3], label='LQR')
    plt.plot(results_ddpg['times'], results_ddpg['states'][:, 3], label='DDPG')
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
    plt.plot(results_ddpg['times'][:-1], results_ddpg['actions'], label='DDPG Control')
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

def load_checkpoint(agent: Union[DDPGAgent, GruDDPGAgent], save_dir):
        # 是否加载先前的训练模型
    checkpoint_files = find_checkpoint_files(save_dir)
    
    load_previous_model = False
    if checkpoint_files: load_previous_model = input("是否加载先前的训练模型? (y/n): ").strip().lower() == 'y' or ''

    start_episode = 0
    previous_model_path = None
    loaded_model_name = None
    episode_rewards = []

    if load_previous_model:
        logging.info(f"准备加载检查点文件")
        
        if checkpoint_files:
            print("\n找到以下检查点文件:")
            for i, file in enumerate(checkpoint_files):
                file_name = os.path.basename(file)
                print(f"{i+1}. {file_name}")
            
            choice = input("请选择要加载的检查点文件编号 (输入数字，直接回车取最新): ")
            
            if choice.strip():
                selected_index = int(choice) - 1
                if 0 <= selected_index < len(checkpoint_files):
                    previous_model_path = checkpoint_files[selected_index]
            else:
                # 默认选择最新的检查点
                previous_model_path = checkpoint_files[0]
                
            loaded_model_name = os.path.basename(previous_model_path)
            
            print(f"加载模型: {loaded_model_name}")
            logging.info(f"加载检查点: {loaded_model_name}")
            
            # 加载模型并获取训练状态
            episode_rewards, current_episode = agent.load_checkpoint(previous_model_path)
            start_episode = current_episode
            # logging.info(f"继续从第 {start_episode} 轮训练，已完成 {len(episode_rewards)} 轮")
            # print(f"继续从第 {start_episode} 轮训练，已完成 {len(episode_rewards)} 轮")
        else:
            print("未找到可加载的检查点文件")
            logging.info("未找到可加载的检查点")
    agent.model_name ,_ = os.path.splitext(loaded_model_name) if loaded_model_name else (None, None)
    return start_episode, episode_rewards

## 通用的绘图函数模块
def plot_data(figsize=(10, 6), plot_title=None, data_sets=None, x_values=None, colors=None, legends=None, 
              xlabel=None, ylabel=None, xlim=None, ylim=None, log_scale=False, 
              line_styles=None, show_legend=True, legend_loc='upper right', save_path=None, show_grid=False, show=True):
    """
    figsize: 绘制的图像大小，默认为(8, 6)\n
    plot_title:图像的标题，默认为空\n
    data_sets: 你要绘制的数据集（可以是多个数组或列表）。若为空则绘图失败\n
    x_values: x轴的值。如果没有提供，默认使用 0, 1, 2, ... 来作为x轴值。\n
    colors: 设置每一条线的颜色，可以是如 ['r', 'g', 'b'] 这样的列表。\n
    legends: 图例的标签。\n
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
        if isinstance(x_values[0], list):
            x_value = x_values[i]
        else:
            x_value = x_values
        
        if len(x_value) != len(data_sets[i]):
            print("x_values的长度与数据集长度不匹配，绘图失败")
            return
        
        color = colors[i] if colors else None
        label = legends[i] if legends else None
        line_style = line_styles[i] if line_styles else '-'
        plt.plot(x_value, data, color=color, label=label, linestyle=line_style)

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
        if plot_title: plt.savefig(save_path +'\\'+ plot_title)
        else: plt.savefig(save_path +'\\'+ 'plot')
    
    if show:
        plt.show()
    else:
        plt.close()
        
def plot_test_data(save_plot_path:str, data, show:bool=True, name:str='',nc_data=None):
    """
    绘制测试数据的函数\n
    只接受一个数据集，数据集的格式为字典\n"""
    x_datas = [data['all_states'][:, 3], nc_data['all_states'][:, 3]] if nc_data else [data['all_states'][:, 3]]
    x_legends = ["无控制"] if not nc_data else ["有控制", "无控制"]
    plot_data(plot_title=f"{name}位移",
          xlabel="时间 (s)",
          ylabel="状态",
          x_values=[data['times'], nc_data['times']] if nc_data else data['times'],
          data_sets=x_datas,
          save_path=save_plot_path,
          legends=x_legends,
          show = show
          )

    plot_data(plot_title=f"{name}动作",
          xlabel="时间 (s)",
          ylabel="动作",
          x_values=data['times'],
          data_sets=[data['actions']],
          save_path=save_plot_path,
          legends=["动作"],
          show = show
          )

    plot_data(plot_title=f"{name}奖励",
          xlabel="时间 (s)",
          ylabel="奖励",
          x_values=data['times'],
          data_sets=[data['rewards']],
          save_path=save_plot_path,
          legends=["奖励"],
          show = show
          )