# 数据驱动的非线性振动系统辨识与控制研究 - 代码复现

本项目复现了论文《数据驱动的非线性振动系统辨识与控制研究》前三章的主要工作。

## 项目结构

```
├── requirements.txt                    # 依赖包列表
├── chapter2/                          # 第2章：基于强化学习的van der Pol振动系统最优控制
│   ├── van_der_pol_system.py         # Van der Pol振动系统模型
│   ├── lqr_controller.py             # LQR控制器
│   ├── td3_controller.py             # TD3强化学习控制器
│   ├── sac_controller.py             # SAC强化学习控制器
│   ├── train_controllers.py          # 控制器训练脚本
│   └── test_controllers.py           # 控制器测试脚本
├── chapter3/                         # 第3章：主动准零刚度隔振器辨识与控制
│   ├── qzs_system.py                 # QZS隔振器动力学模型
│   ├── rk4_pinn.py                   # RK4-PINN代理模型辨识
│   ├── pso_pid_controller.py         # PSO优化的PID控制器
│   ├── train_qzs_controllers.py      # QZS控制器训练脚本
│   └── test_qzs_performance.py       # QZS隔振性能测试
├── chapter4/                         # 第4章：多级非线性隔振系统神经网络模型预测控制
│   ├── multilayer_system.py          # 多级非线性隔振系统模型
│   ├── substructure_pinn.py          # 子结构物理信息神经网络
│   ├── nn_mpc_controller.py          # 神经网络模型预测控制器
│   ├── train_multilayer_controllers.py # 多级系统控制器训练
│   └── test_multilayer_performance.py  # 多级系统性能测试
├── utils/                             # 通用工具函数
│   ├── __init__.py
│   ├── plotting.py                   # 绘图工具
│   ├── data_utils.py                 # 数据处理工具
│   └── math_utils.py                 # 数学计算工具
└── README.md                         # 项目说明
```

## 主要功能

### 第2章：Van der Pol振动系统最优控制
- Van der Pol非线性振动系统建模
- LQR最优控制器设计
- TD3和SAC深度强化学习控制器训练
- 控制性能比较分析

### 第3章：主动QZS隔振器辨识与控制
- QZS隔振器动力学建模
- 基于RK4-PINN的代理模型辨识
- PSO-PID和TD3控制器优化
- 变负载工况下的隔振性能分析

### 第4章：多级非线性隔振系统控制
- 多级非线性隔振系统建模
- 子结构物理信息神经网络方法
- 神经网络模型预测控制器设计
- 隔振性能测试和分析

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行第2章代码：
```bash
cd chapter2
python train_controllers.py  # 训练控制器
python test_controllers.py   # 测试控制器性能
```

3. 运行第3章代码：
```bash
cd chapter3
python train_qzs_controllers.py  # 训练QZS控制器
python test_qzs_performance.py   # 测试QZS隔振性能
```

4. 运行第4章代码：
```bash
cd chapter4
python train_multilayer_controllers.py  # 训练多级系统控制器
python test_multilayer_performance.py   # 测试多级系统性能
```

## 论文引用

蒋纪元. 数据驱动的非线性振动系统辨识与控制研究[D]. 西南交通大学, 2024.
