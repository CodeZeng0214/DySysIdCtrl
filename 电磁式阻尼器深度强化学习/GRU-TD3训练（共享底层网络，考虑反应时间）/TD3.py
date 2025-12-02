## TD3 算法定义的函数

import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, List
from nn import Actor, Critic, GruPredictor_norm, ReplayBuffer, Gru_Actor, Gru_Critic, Gru_ReplayBuffer, GruPredictor_diff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## TD3代理基类
class BaseTD3Agent:
    def __init__(self, state_dim=1, action_dim=1, mlp_hidden_dim=128, gru_hidden_dim=64, action_bound=5.0, 
                 actor_lr=1e-3, critic_lr=1e-3, clip_grad=False, gamma=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, action_sigma=0.2,
                 aware_dt: bool = False, aware_delay_time: bool = False,
                 delay_enabled: bool = False, delay_step: int = 5, delay_sigma: int = 2):
        """初始化TD3代理\n
        - state_dim 状态维度
        - action_dim 动作维度
        - mlp_hidden_dim MLP隐藏层维度
        - gru_hidden_dim GRU隐藏层维度
        - action_bound 动作范围
        - actor_lr Actor学习率
        - critic_lr Critic学习率
        - clip_grad 是否使用梯度裁剪
        - gamma 折扣因子
        - tau 软更新参数
        - policy_noise 目标策略平滑正则化噪声
        - noise_clip 噪声裁剪范围
        - policy_freq 策略更新频率
        - action_sigma 探索最大噪声
        - aware_dt 是否使用时间步长作为状态的一部分
        - aware_delay_time 是否启用延迟感知
        - delay_enabled 是否启用动作延迟
        - delay_step 延迟步数
        - delay_sigma 延迟步数的标准差
        """
        # 初始化参数
        self.freeze_gru = False  # 是否冻结GRU参数
        
        self.state_dim = state_dim # 状态维度
        self.action_dim = action_dim # 动作维度
        self.mlp_hidden_dim = mlp_hidden_dim # mlp隐藏层维度
        self.gru_hidden_dim = gru_hidden_dim # gru隐藏层维度
        self.action_bound = action_bound # 动作范围
        
        self.clip_grad = clip_grad # 是否使用梯度裁剪
        self.actor_lr = actor_lr # Actor网络学习率
        self.critic_lr = critic_lr # Critic网络学习率
        self.gamma = gamma # 折扣因子
        self.tau = tau # 软更新参数
        
        self.policy_noise = policy_noise # 目标策略平滑正则化噪声
        self.noise_clip = noise_clip # 噪声裁剪范围
        self.policy_freq = policy_freq # 策略更新频率
        self.action_sigma = action_sigma # 探索最大噪声
        
        self.aware_dt = aware_dt # 是否使用时间步长作为状态的一部分
        self.aware_delay_time = aware_delay_time # 是否启用延迟感知
        self.delay_enabled = delay_enabled # 是否启用动作延迟
        self.delay_step = delay_step # 延迟步数
        self.delay_sigma = delay_sigma # 延迟步数的标准差
        
        self.model_name = None
        self.total_it = 0 # 总迭代次数
        self.episode_rewards = [] # 存储每个回合的奖励

    def _init_nn(self):
        # 需要在子类中定义
        self.actor: Actor | Gru_Actor = None
        self.critic1: Critic | Gru_Critic = None
        self.critic2: Critic | Gru_Critic = None
        self.target_actor: Actor | Gru_Actor = None
        self.target_critic1: Critic | Gru_Critic = None
        self.target_critic2: Critic | Gru_Critic = None
        self.gru_predictor: GruPredictor_diff = None
        self.target_gru_predictor: GruPredictor_diff = None
        raise NotImplementedError("需要在子类中初始化神经网络结构")

    def _init_optimizer(self):
        # 优化器设置
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        
    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def select_action(self):
        """选择动作"""
        raise NotImplementedError("需要在子类中实现动作选择方法")
            
    def reset_history(self):
        """重置状态历史，在新的episode开始时调用"""        
        if hasattr(self, 'state_history'):
            self.state_history = []
        
    def update(self, replay_buffer: Union[ReplayBuffer, Gru_ReplayBuffer]) -> Tuple[float, float, float]:
        """更新Actor和Critic网络"""
        if len(replay_buffer) < replay_buffer.batch_size:
            return 0.0, 0.0, 0.0
        
        self.total_it += 1
        
        # 1. 从回放池中采样
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        
        with torch.no_grad():
            # 目标策略平滑正则化
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor.forward(next_states) + noise).clamp(-self.action_bound, self.action_bound)
            
            # 计算目标Q值，取两个Critic的最小值
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + self.gamma * target_q * (1 - dones)
        
        self._freeze_gru()
        
        # 2. 更新两个Critic网络
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        if self.total_it % 100 == 0: self.check_grad(self.critic1,threshold_high=int(self.clip_grad) if self.clip_grad else 10.0)
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=int(self.clip_grad) if self.clip_grad else 10.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.total_it % 100 == 0: self.check_grad(self.critic2,threshold_high=int(self.clip_grad) if self.clip_grad else 10.0)
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=int(self.clip_grad) if self.clip_grad else 10.0)
        self.critic2_optimizer.step()
        
        critic_loss = (critic1_loss + critic2_loss) / 2
        
        actor_loss = 0.0
        # 3. 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            # 更新Actor网络
            policy_actions = self.actor(states)
            actor_loss = -self.critic1.forward(states, policy_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # 打印梯度信息（调试用）
            if self.total_it % 100 == 0: self.check_grad(self.actor,threshold_high=int(self.clip_grad) if self.clip_grad else 10.0)
            
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=int(self.clip_grad) if self.clip_grad else 10.0)
            
            # # 冻结gru_predictor共享的参数
            # if self.gru_predictor is not None: self.gru_predictor.freeze_gru()
            
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)
            
            actor_loss = actor_loss.item()
        
        return critic_loss.item(), actor_loss, (critic1_loss.item() + critic2_loss.item()) / 2
  
    def _freeze_gru(self):
        if self.gru_predictor is not None and self.freeze_gru:
            self.actor.gru_predictor.freeze_gru()
            self.target_actor.gru_predictor.freeze_gru()
            self.critic1.gru_predictor.freeze_gru()
            self.target_critic1.gru_predictor.freeze_gru()
            self.critic2.gru_predictor.freeze_gru()
            self.target_critic2.gru_predictor.freeze_gru()
        else:
            pass

    
    def check_grad(self, model: torch.nn.Module, verbose=False, threshold_high=10.0, threshold_low=1e-8):
        """检查模型的梯度情况，支持迭代检查各层
        
        Args:
            model: 要检查的模型
            verbose: 是否打印详细的每层梯度信息
            threshold_high: 梯度过高的阈值
            threshold_low: 梯度过低的阈值
        """
        total_grad_norm = 0
        param_count = 0
        layer_stats = {}  # 存储各层统计信息
        
        # 1. 迭代检查各层梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                param_count += 1
                
                # 提取层名（去掉参数类型后缀）
                layer_name = '.'.join(name.split('.')[:-1]) if '.' in name else name
                
                # 统计各层梯度
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {
                        'grad_norms': [],
                        'param_names': [],
                        'param_shapes': []
                    }
                
                layer_stats[layer_name]['grad_norms'].append(grad_norm)
                layer_stats[layer_name]['param_names'].append(name)
                layer_stats[layer_name]['param_shapes'].append(tuple(param.shape))
                
                # 详细模式：打印每个参数的梯度
                if verbose:
                    print(f"  📍 {name}: shape={param.shape}, grad_norm={grad_norm:.6f}")
                    logging.info(f"  📍 {name}: shape={param.shape}, grad_norm={grad_norm:.6f}")
        
        total_grad_norm = total_grad_norm ** 0.5
        
        # 2. 打印总体统计
        # print(f"\n🔍 本轮第{self.total_it}次更新， {model.__name__} 梯度检查报告:")
        logging.info(f"🔍 本轮第{self.total_it}次更新， {model.__class__.__name__} 梯度检查报告:")
        # print(f"  └─总梯度范数: {total_grad_norm:.6f}")
        logging.info(f"  └─ 总梯度范数: {total_grad_norm:.6f}")

        # 3. 打印各层统计
        for layer_name, stats in layer_stats.items():
            grad_norms = stats['grad_norms']
            avg_grad = np.mean(grad_norms)
            max_grad = np.max(grad_norms)
            min_grad = np.min(grad_norms)
            
            # 判断异常状态
            status = "✅"
            if max_grad > threshold_high:
                status = "🔴 过高"
            elif max_grad < threshold_low:
                status = "⚪ 过低"
            
            # print(f"  {status} {layer_name}:")
            # print(f"      ├─ 平均梯度: {avg_grad:.6e}")
            # print(f"      ├─ 最大梯度: {max_grad:.6e}")
            # print(f"      ├─ 最小梯度: {min_grad:.6e}")
            # print(f"      └─ 参数: {stats['param_names']}")
            if verbose:
                # print("\n📊 各层梯度统计:")
                logging.info("📊 各层梯度统计:")
                logging.info(f"{status} {layer_name} , 参数: {stats['param_names']} : ")
                logging.info(f"- 平均: {avg_grad:.6e}, 最大: {max_grad:.6e}, 最小: {min_grad:.6e}")
            
        # 4. 检查梯度是否异常
        # print("\n⚠️  异常检测:")
        if total_grad_norm > threshold_high:
            msg = f"❌❌❌ 梯度爆炸! 总梯度范数: {total_grad_norm:.6f} (阈值: {threshold_high})"
            # print(msg)
            logging.warning(msg)
            
            # 找出梯度最大的层
            max_layer = max(layer_stats.items(), key=lambda x: np.max(x[1]['grad_norms']))
            # print(f"   └─ 最大梯度来自: {max_layer[0]} (梯度范数: {np.max(max_layer[1]['grad_norms']):.6f})")
            logging.warning(f"   └─ 最大梯度来自: {max_layer[0]} (梯度范数: {np.max(max_layer[1]['grad_norms']):.6f})")
            
        elif total_grad_norm < threshold_low:
            msg = f"❌❌❌ 梯度消失! 总梯度范数: {total_grad_norm:.6e} (阈值: {threshold_low})"
            # print(msg)
            logging.warning(msg)
        else:
            # print("✅ 梯度正常")
            pass
            logging.info("✅ 梯度正常")

        # print("-" * 60)
        logging.info("-" * 60)
        
        return {
            'total_grad_norm': total_grad_norm,
            'param_count': param_count,
            'layer_stats': layer_stats
        }
    
## TD3 代理
class TD3Agent(BaseTD3Agent):
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0,
                 actor_lr=1e-3, critic_lr=1e-3, clip_grad=False, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, action_sigma=0.2,
                 aware_dt: bool = False, aware_delay_time: bool = False,
                 delay_enabled: bool = False, delay_step: int = 5, delay_sigma: int = 2):
        # 初始化参数
        super().__init__(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, action_bound=action_bound,
                 actor_lr=actor_lr, critic_lr=critic_lr, clip_grad=clip_grad, gamma=gamma, tau=tau,
                 policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq, action_sigma=action_sigma,
                 aware_dt=aware_dt, aware_delay_time=aware_delay_time,
                 delay_enabled=delay_enabled, delay_step=delay_step, delay_sigma=delay_sigma)
        self._init_nn()
        self._init_optimizer()

    def _init_nn(self):
        # 网络初始化
        self.actor = Actor(self.state_dim, self.action_dim, self.mlp_hidden_dim, self.action_bound).to(device)
        self.critic1 = Critic(self.state_dim, self.action_dim, self.mlp_hidden_dim).to(device)
        self.critic2 = Critic(self.state_dim, self.action_dim, self.mlp_hidden_dim).to(device)
        
        self.target_actor = Actor(self.state_dim, self.action_dim, self.mlp_hidden_dim, self.action_bound).to(device)
        self.target_critic1 = Critic(self.state_dim, self.action_dim, self.mlp_hidden_dim).to(device)
        self.target_critic2 = Critic(self.state_dim, self.action_dim, self.mlp_hidden_dim).to(device)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def select_action(self, state_history: List[np.ndarray], add_noise=True, epsilon=1.0, rand_prob=0.05, delay=1) -> float:
        """选择动作，支持探索"""
        # 如果启用动作延迟，使用延迟步数的高斯分布采样
        if self.delay_enabled:
            if len(state_history) < delay:
                state = state_history[-1]  # 如果历史长度不够，使用最新状态
            else:
                state = state_history[-delay]  # 使用延迟的状态
                
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_tensor: torch.Tensor = self.actor(state)
            action_np: np.ndarray = action_tensor.cpu().detach().numpy()
            action = action_np.flatten()
            
        if add_noise:
            noise = np.random.normal(0, self.action_bound * self.action_sigma * epsilon, size=self.action_dim)
            action += noise
            if np.random.random() < rand_prob:
                action = np.random.uniform(-self.action_bound, self.action_bound, self.action_dim)

        return float(np.clip(action, -self.action_bound, self.action_bound))
        
## 基于GRU的TD3代理（分离式架构）
class Gru_TD3Agent(BaseTD3Agent):
    def __init__(self, norm: bool = False, simple_nn: bool = False,freeze_gru: bool = False,
                 state_dim=1, action_dim=1, mlp_hidden_dim=128, gru_hidden_dim=64, action_bound=5.0,
                 actor_lr=5e-4, critic_lr=1e-3, gru_predictor_lr=1e-3, clip_grad=False, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, action_sigma=0.2, 
                 aware_dt: bool = False, aware_delay_time: bool = False,
                 delay_enabled: bool = False, delay_step: int = 5, delay_sigma: int = 2,
                 seq_len=10, gru_layers=1, fc_seq_len=1):
        """初始化GRU-TD3代理（分离式架构）
        
        新增参数：
        - gru_predictor_lr: GRU预测器学习率
        """
        super().__init__(state_dim=state_dim, action_dim=action_dim, mlp_hidden_dim=mlp_hidden_dim, gru_hidden_dim=gru_hidden_dim, action_bound=action_bound,
                 actor_lr=actor_lr, critic_lr=critic_lr, clip_grad=clip_grad, gamma=gamma, tau=tau,
                 policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq, action_sigma=action_sigma,
                 aware_dt=aware_dt, aware_delay_time=aware_delay_time,
                 delay_enabled=delay_enabled, delay_step=delay_step, delay_sigma=delay_sigma)

        self.gru_layers = gru_layers  # GRU层数
        self.seq_len = seq_len  # 序列长度
        self.fc_seq_len = fc_seq_len  # 预测时间步长度
        self.gru_predictor_lr = gru_predictor_lr  # GRU预测器学习率
        self.freeze_gru = freeze_gru  # 是否冻结GRU参数

        self._init_nn(norm=norm, simple_nn=simple_nn)
        self._init_optimizer()
    
    def _init_nn(self, norm: bool = False, simple_nn: bool = False):
        # 创建共享的GRU预测器
        # gru_state_dim = 2 + int(self.aware_dt) + int(self.aware_delay_time)  # 状态维度 + 时间步长 + 延迟时间感知
        gru_state_dim = self.state_dim
        self.gru_predictor = GruPredictor_norm(norm=norm, simple_nn=simple_nn,freeze_gru=self.freeze_gru,
            state_dim=gru_state_dim, hidden_dim=self.gru_hidden_dim, num_layers=self.gru_layers, fc_seq_len=self.fc_seq_len,
            aware_dt=self.aware_dt, aware_delay_time=self.aware_delay_time
            ).to(device)
        self.gru_predictor1 = GruPredictor_norm(norm=norm, simple_nn=simple_nn,freeze_gru=self.freeze_gru,
            state_dim=gru_state_dim, hidden_dim=self.gru_hidden_dim, num_layers=self.gru_layers, fc_seq_len=self.fc_seq_len,
            aware_dt=self.aware_dt, aware_delay_time=self.aware_delay_time
            ).to(device)
        self.gru_predictor2 = GruPredictor_norm(norm=norm, simple_nn=simple_nn,freeze_gru=self.freeze_gru,
            state_dim=gru_state_dim, hidden_dim=self.gru_hidden_dim, num_layers=self.gru_layers, fc_seq_len=self.fc_seq_len,
            aware_dt=self.aware_dt, aware_delay_time=self.aware_delay_time
            ).to(device)
        self.target_gru_predictor = GruPredictor_norm(norm=norm, simple_nn=simple_nn,freeze_gru=self.freeze_gru,
            state_dim=gru_state_dim, hidden_dim=self.gru_hidden_dim, num_layers=self.gru_layers, fc_seq_len=self.fc_seq_len,
            aware_dt=self.aware_dt, aware_delay_time=self.aware_delay_time
            ).to(device)
        self.target_gru_predictor1 = GruPredictor_norm(norm=norm, simple_nn=simple_nn,freeze_gru=self.freeze_gru,
            state_dim=gru_state_dim, hidden_dim=self.gru_hidden_dim, num_layers=self.gru_layers, fc_seq_len=self.fc_seq_len,
            aware_dt=self.aware_dt, aware_delay_time=self.aware_delay_time
            ).to(device)
        self.target_gru_predictor2 = GruPredictor_norm(norm=norm, simple_nn=simple_nn,freeze_gru=self.freeze_gru,
            state_dim=gru_state_dim, hidden_dim=self.gru_hidden_dim, num_layers=self.gru_layers, fc_seq_len=self.fc_seq_len,
            aware_dt=self.aware_dt, aware_delay_time=self.aware_delay_time
            ).to(device)

        # GRU网络初始化（传入共享的GRU预测器）
        self.actor = Gru_Actor(self.gru_predictor, norm=norm, simple_nn=simple_nn,
                               state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.mlp_hidden_dim, action_bound=self.action_bound).to(device)

        self.critic1 = Gru_Critic(self.gru_predictor1, norm=norm, simple_nn=simple_nn,
                                  state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.mlp_hidden_dim, gru_hidden_dim=self.gru_hidden_dim).to(device)

        self.critic2 = Gru_Critic(self.gru_predictor2, norm=norm, simple_nn=simple_nn,
                                  state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.mlp_hidden_dim, gru_hidden_dim=self.gru_hidden_dim).to(device)

        # 目标网络使用目标GRU预测器
        self.target_actor = Gru_Actor(self.target_gru_predictor, norm=norm, simple_nn=simple_nn,
                                       state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.mlp_hidden_dim, action_bound=self.action_bound).to(device)

        self.target_critic1 = Gru_Critic(self.target_gru_predictor1, norm=norm, simple_nn=simple_nn,
                                         state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.mlp_hidden_dim, gru_hidden_dim=self.gru_hidden_dim).to(device)

        self.target_critic2 = Gru_Critic(self.target_gru_predictor2, norm=norm, simple_nn=simple_nn,
                                         state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.mlp_hidden_dim, gru_hidden_dim=self.gru_hidden_dim).to(device)

        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.modelnn: List[torch.nn.Module] = [self.actor, self.critic1, self.critic2, self.target_actor, self.target_critic1, self.target_critic2,
                        self.gru_predictor, self.target_gru_predictor, self.gru_predictor1, self.target_gru_predictor1, self.gru_predictor2, self.target_gru_predictor2]

    def _init_optimizer(self):
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        
        # ✅ 修正：统计参数元素数量，而非参数对象数量
        actor_param_count = sum(p.numel() for p in self.actor.parameters())
        gru_param_count = sum(p.numel() for p in self.gru_predictor.parameters())
        
        # 验证GRU参数是否被包含
        actor_param_ids = set(id(p) for p in self.actor.parameters())
        gru_param_ids = set(id(p) for p in self.gru_predictor.parameters())
        
        if not gru_param_ids.issubset(actor_param_ids):
            raise ValueError("❌ GRU预测器参数未包含在Actor优化器中！")
        
        print(f"✅ Actor网络总参数数量: {actor_param_count:,}")
        logging.info(f"✅ Actor网络总参数数量: {actor_param_count:,}")
        print(f"✅ 其中GRU预测器参数数量: {gru_param_count:,}")
        logging.info(f"✅ 其中GRU预测器参数数量: {gru_param_count:,}")
        print(f"✅ Actor优化器包含 {len(actor_param_ids)} 个参数对象")
        logging.info(f"✅ Actor优化器包含 {len(actor_param_ids)} 个参数对象")
        print(f"✅ 其中GRU预测器 {len(gru_param_ids)} 个参数对象")
        logging.info(f"✅ 其中GRU预测器 {len(gru_param_ids)} 个参数对象")
        
        # 详细打印各层参数
        print("\n📊 Actor网络参数详情:")
        logging.info("📊 Actor网络参数详情:")
        for name, param in self.actor.named_parameters():
            print(f"  - {name}: {param.shape} ({param.numel():,} 个元素)")
            logging.info(f"  - {name}: {param.shape} ({param.numel():,} 个元素)")
        
        print("\n📊 GRU预测器参数详情:")
        logging.info("📊 GRU预测器参数详情:")
        for name, param in self.gru_predictor.named_parameters():
            print(f"  - {name}: {param.shape} ({param.numel():,} 个元素)")
            logging.info(f"  - {name}: {param.shape} ({param.numel():,} 个元素)")
    
    def select_action(self, state_history: List[np.ndarray], add_noise=True, epsilon=1.0, rand_prob=0.05, delay=1) -> float:
        """选择动作，支持探索"""

        # 如果历史长度不够，使用零填充或重复当前状态
        required_len = self.seq_len + delay - 1
        if len(state_history) < required_len:
            # 用当前状态填充不足的部分
            padding_len = required_len - len(state_history)
            padded_history = [state_history[0]] * padding_len + list(state_history)
        else:
            padded_history = list(state_history)
        
        # 取延迟后的序列
        if delay == 1:
            state_seq = padded_history[-self.seq_len:]
        else:
            state_seq = padded_history[-self.seq_len-delay+1:-delay+1]

        state_seq_tensor = torch.tensor(np.array(state_seq), dtype=torch.float32).unsqueeze(0).to(device)  # [1, seq_len, state_dim]
        
        with torch.no_grad():
            action_tensor: torch.Tensor = self.actor(state_seq_tensor)
            action_np: np.ndarray = action_tensor.cpu().detach().numpy()
            action = action_np.flatten()
            
        if add_noise:
            noise = np.random.normal(0, self.action_bound * self.action_sigma * epsilon, size=self.action_dim)
            action += noise
            if np.random.random() < rand_prob:
                action = np.random.uniform(-self.action_bound, self.action_bound, self.action_dim)

        return float(np.clip(action, -self.action_bound, self.action_bound))
    
        # with torch.no_grad():
        #     state = state_seq_tensor
        #     action_tensor: torch.Tensor = self.actor(state)
        #     action_np: np.ndarray = action_tensor.cpu().detach().numpy()
        #     action = action_np.flatten()
            
        # if add_noise:
        #     noise = np.random.normal(0, self.action_sigma * epsilon, size=self.action_dim)
        #     action += noise
        #     if np.random.random() < rand_prob:
        #         action = np.random.uniform(-self.action_bound, self.action_bound, self.action_dim)

        # return float(np.clip(action, -1, 1))