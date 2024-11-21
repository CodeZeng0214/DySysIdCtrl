### 2024.11.14
## 此代码实现的是论文 蒋纪元《数据驱动的动力学系统建模及控制策略研究》的 3.3节 基于神经网络的系统辨识 
# 通过循环神经网络RNN逼近动态系统
# 基于FNN的构建思路构建了 神经网络对输入的响应及训练损失的方法框架 里面调用自己定制的训练、仿真、绘图等函数

#@ 待完善的内容是将FNN的网络并入此框架中，最好能将代码写为class类的方式
#@ （已在DDPG算法中尝试）另一个需要完善的是把绘图等函数包装为一个通用的class类


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split


# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


## 生成真实的响应序列
def Response_true(which_u):
    y = np.zeros(n_samples)
    if which_u == 1:
        #u(t) = 0.5 sin( 0.6πt)
        u = 0.5 * np.sin(0.6 * np.pi * np.arange(0, T, dt))
    else:    
        #u(t) = 0.4 cos( 0.4πt) + 0.15 sin( 3πt)
        u = 0.4 * np.sin(0.4 * np.pi * np.arange(0, T, dt)) + 0.15 * np.sin(3 * np.pi * np.arange(0, T, dt))
    # 生成真实的输出序列
    for k in range(1, n_samples):
        y[k] = u[k]**3 + y[k-1] / (1 + y[k-1]**2)
    return u,y


## 循环神经网络RNN具体架构
class RNN1(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=2):
        super(RNN1, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(2, x.size(0), 32).to(x.device)
        #print("x shape:", x.shape)
        #print("h0 shape:", h0.shape)
        # RNN 层的输出 `out` 形状为 [batch_size, sequence_length, hidden_size]
        out, _ = self.rnn(x, h0)
        
        # 应用全连接层到每个时间步的输出
        out = self.fc(out)  # 形状变为 [batch_size, sequence_length, output_size]
        
        return out

class RNN2(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1, num_layers=2):
        super(RNN2, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(2, x.size(0), 32).to(x.device)
        
        # RNN 层的输出 `out` 形状为 [batch_size, sequence_length, hidden_size]
        out, _ = self.rnn(x, h0)
        
    # 应用全连接层到每个时间步的输出
        out = self.fc(out)  # 形状变为 [batch_size, sequence_length, output_size]
        return out


## 准备数据集
def DataProcess(u, y_true, which_model):
    u_input = np.concatenate((np.zeros(sequence_length, dtype=int), u)) # 为实现序列数据预测所有时间，因此将u和y_true扩展sequence_length个零
    y_input = np.concatenate((np.zeros(sequence_length, dtype=int), y_true))
    y_samples = [y_input[i:i + sequence_length] for i in range(len(y_input) - sequence_length)]
    u_samples = [u_input[i:i + sequence_length] for i in range(len(u_input) - sequence_length)]
    y_samples = np.array(y_samples)
    u_samples = np.array(u_samples)
    if which_model == 1:
        X = u_samples
    else:
        # 使用 np.hstack 合并
        X = [np.hstack((u.reshape(-1, 1), y.reshape(-1, 1))) for u, y in zip(u_samples, y_samples)]
    Y = y_samples
    return X, Y


## RNN训练函数
def RNN_Train(X,Y,which_model=1,which_u=1):
    
    #根据输入初始化参数
    input_size = X[0].ndim
    if which_model==1: model = RNN1() 
    else: model = RNN2()
    sign = f'RNN{which_model}_u{which_u}'
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 转换数据为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, sequence_length, input_size)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, sequence_length,1)
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, sequence_length, input_size)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, sequence_length,1)
    # 模型参数设定
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_data = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # 初始化损失记录
    train_losses = []
    test_losses = []
    
    # CUDA 加速
    model = model.to('cuda')

    for epoch in range(max_epochs):
        # 训练模式
        model.train()
        batch_losses = []
        for inputs, targets in train_loader:
            
            inputs = inputs.to('cuda')
            targets = targets.to('cuda')
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 检查数据维度是否匹配
            #print("Input shape:", inputs.shape)
            # print("Target shape:", targets.shape)
            # print("Output shape:", outputs.shape)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        # 保存每个 epoch 的平均训练损失
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # 计算测试集损失
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)  # 将测试集输入数据移动到GPU
            Y_test = Y_test.to(device)  # 将测试集目标数据移动到GPU
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, Y_test).item()
            test_losses.append(test_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{max_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    # 保存模型
    torch.save(model.state_dict(), f"./Identify_Solution/Neural_Network/{sign}_model.pth")
    
    #保存损失函数
    np.save(f'./Identify_Solution/Neural_Network/{sign}_train_losses.npy', np.array(train_losses))
    np.save(f'./Identify_Solution/Neural_Network/{sign}_test_losses.npy', np.array(test_losses))
        
    return train_losses, test_losses


## 绘制损失曲线(不打开图窗)
def Loss_plot(train_losses1,test_losses1,train_losses2,test_losses2,which_u):
    sign = f'RNN_u{which_u}'
    plt.plot(train_losses1, label="RNN1 Train Loss", color="blue")
    plt.plot(test_losses1, label="RNN1 Test Loss", color="blue", linestyle="--")
    plt.plot(train_losses2, label="RNN2 Train Loss", color="red")
    plt.plot(test_losses2, label="RNN2 Test Loss", color="red", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")  # 设置纵坐标为对数刻度
    plt.legend()
    plt.title(f"{sign}_Training and Testing Loss over Epochs")
    plt.savefig(f"./Identify_Solution/Neural_Network/{sign}_Loss.png", dpi=300, bbox_inches='tight')
    

## 利用RNN模型预测数据
def RNN_Response(which_model, which_u, X_input):
# 使用神经网络进行仿真
    sign = f'RNN{which_model}_u{which_u}'
    if which_model==1: rnn = RNN1() 
    else: rnn = RNN2()
    model = rnn  # 定义 FNN 结构
    model.load_state_dict(torch.load(f"./Identify_Solution/Neural_Network/{sign}_model.pth"))
    model.eval()
    model = model.to('cuda')
    y_nn = np.zeros(n_samples)
    y_nn_samples = [np.zeros(sequence_length) for _ in range(n_samples)]
    with torch.no_grad():
        for k in range(1, n_samples):
            # if which_model == 1:
            #     X_input = X_input[k]
            # else:
            # # 使用 np.hstack 合并
            #     X_input = [np.hstack((u_samples[k].reshape(-1, 1), y_samples[k-1].reshape(-1, 1)))]
            # 输入跟随网络结构变化
            input_data = torch.tensor(X_input[k], dtype=torch.float32).view(1,sequence_length, X_input[0].ndim).to(device)
            y_nn_samples[k] = model(input_data).cpu().numpy().flatten()
            y_nn[k] = y_nn_samples[k][-1]
    return y_nn


## 绘制预测与真实响应的对比图
def Resultscontrast_Plot(y_true,y_nn1,y_nn2,which_u):
    
    sign = f'u{which_u}'
    plt.plot(time,y_true[:n_samples], label="True Output", color="blue")
    plt.plot(time,y_nn1[:n_samples], label="RNN1 Output", color="red", linestyle="--")
    plt.plot(time,y_nn2[:n_samples], label="RNN2 Output", color="green", linestyle="--")
    plt.xlim(0, T)  # 将横坐标限制在 0 到 T
    plt.xlabel("T/s")
    plt.ylabel("Output y[k]")
    plt.title(f"{sign}_System Output: True vs RNN1 vs RNN2")
    plt.legend()
    plt.legend(loc="upper right") # 图例放在右上角
    plt.savefig(f"./Identify_Solution/Neural_Network/RNN_TRUE_Resultscontrast_{sign}.png", dpi=300, bbox_inches='tight')
    

## 绘制跟踪误差图
def Error_plot(y_true,y_nn1,y_nn2,which_u):

    sign = f'u{which_u}'
    error_nn1 = y_true[:n_samples] - y_nn1[:n_samples]
    error_nn2 = y_true[:n_samples] - y_nn2[:n_samples]
    plt.figure(figsize=(12, 6))
    plt.plot(time,error_nn1, label="RNN1 Error", color="green")
    plt.plot(time,error_nn2, label="RNN2 Error", color="red")
    plt.xlim(0, T)  # 将横坐标限制在 0 到 T
    plt.xlabel("T/s")
    plt.ylabel("Tracking Error")
    plt.title(f"{sign}_Tracking Error: RNN1 vs RNN2")
    plt.legend()
    plt.legend(loc="upper right") # 图例放在右上角
    plt.savefig(f"./Identify_Solution/Neural_Network/RNN_TrackError_{sign}.png", dpi=300, bbox_inches='tight')


## 获取神经网络对输入的响应及训练损失的方法框架
def MethodFrame(which_u,which_model,istrain):
    
    sign = f'RNN{which_model}_u{which_u}'
    
    # 获取真实响应
    u, y_true = Response_true(which_u=which_u)
    
    # 处理数据集
    X, Y = DataProcess(u, y_true, which_model=which_model)
    
    # 是否需要训练    
    if istrain:
        train_losses, test_losses = RNN_Train(X, Y, which_model=which_model,which_u=which_u)
    
    # 获取神经网络对输入的响应
    y_nn = RNN_Response(which_model=which_model,which_u=which_u,X_input=X)
    
    # 返回需要的变量
    if not istrain:
        train_losses = np.load(f'./Identify_Solution/Neural_Network/{sign}_train_losses.npy')
        test_losses = np.load(f'./Identify_Solution/Neural_Network/{sign}_test_losses.npy')
    return y_true, train_losses, test_losses, y_nn
        


# 系统参数
dt = 0.01  # Step size
T = 20    # Simulation duration
time = np.arange(0, T , dt)
n_samples = time.size

# 训练参数
learning_rate = 1e-5 # 学习率
batch_size = 4 # 批量学习大小
max_epochs = 200 # 最大迭代轮次
sequence_length = 100 # 切割数据的长度
istrain = False # 是否为训练模式


############################# 输入u1
which_u = 1
# 输入u1，模型RNN1
y_true1, train_losses11, test_losses11, y_nn11 = MethodFrame(which_u=which_u,which_model=1,istrain=istrain)
# 输入u1，模型RNN2
y_true1, train_losses12, test_losses12, y_nn12 = MethodFrame(which_u=which_u,which_model=2,istrain=istrain)

## 绘制损失函数对比图
plt.figure(figsize=(12, 6))
Loss_plot(train_losses11,test_losses11,train_losses12,test_losses12,which_u=which_u)

## 绘制仿真结果对比图
plt.figure(figsize=(12, 6))
Resultscontrast_Plot(y_true1,y_nn11,y_nn12,which_u=which_u)
## 绘制跟踪误差图
Error_plot(y_true1,y_nn11,y_nn12,which_u=which_u)
#plt.show()
#############################


############################# 输入u2
which_u = 2
# 输入u2，模型RNN1
y_true2, train_losses21, test_losses21, y_nn21 = MethodFrame(which_u=which_u,which_model=1,istrain=istrain)
# 输入u2，模型RNN2
y_true2, train_losses22, test_losses22, y_nn22 = MethodFrame(which_u=which_u,which_model=2,istrain=istrain)

## 绘制损失函数对比图
plt.figure(figsize=(12, 6))
Loss_plot(train_losses21,test_losses21,train_losses22,test_losses22,which_u=which_u)

## 绘制仿真结果对比图
plt.figure(figsize=(12, 6))
Resultscontrast_Plot(y_true2,y_nn21,y_nn22,which_u=which_u)
## 绘制跟踪误差图
Error_plot(y_true2,y_nn21,y_nn22,which_u=which_u)
plt.show()
#############################
