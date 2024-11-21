### 2024.11.12
## 此代码实现的是论文 蒋纪元《数据驱动的动力学系统建模及控制策略研究》的 3.3节 基于神经网络的系统辨识 
# 通过循环神经网络FNN逼近动态系统

#@（已经于RNN代码里实现部分内容） 优化目标是把代码包装成一个个模块

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


## 前馈神经网络FNN具体架构
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(2, 64)     # Input layer
        self.layer2 = nn.Linear(64, 64)    # Hidden layer 1
        self.layer3 = nn.Linear(64, 64)    # Hidden layer 2
        self.layer4 = nn.Linear(64, 1)     # Output layer

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(64, 1)     # Output layer
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.tanh(x)
        x = self.layer4(x)
        return x
    
## FNN训练函数
def FNN_Train(X,Y,learning_rate,batch_size,max_epochs,sign):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    # Model, Loss, and Optimizer
    model = FNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_data = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 初始化损失记录
    train_losses = []
    test_losses = []

    for epoch in range(max_epochs):
        # 训练模式
        model.train()
        batch_losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        # 保存每个epoch的平均训练损失
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # 计算测试集损失
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, Y_test).item()
            test_losses.append(test_loss)
        
        if epoch % 50 == 0:
            print(f'Epoch [{epoch}/{max_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    torch.save(model.state_dict(), f"./Identify_Solution/Neural_Network/fnn_model{sign}.pth")
            
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")  # 设置纵坐标为对数刻度
    plt.legend()
    plt.title("Training and Testing Loss over Epochs")
    plt.savefig(f"./Identify_Solution/Neural_Network/FNN_TrainLoss{sign}.png", dpi=300, bbox_inches='tight')
    

## 生成真实的响应序列
def Response_true(y,u):
    # 生成真实的输出序列
    for k in range(1, n_samples):
        y[k] = u[k]**3 + y[k-1] / (1 + y[k-1]**2)
    return y

## ARMA预测方法
def ARMA_Predict(y_true,u):
## 拟合 ARMA 模型的参数 a1 和 b0
# 通过最小二乘法拟合模型参数
    # 创建矩阵用于最小二乘拟合
    A = np.vstack([-y_true[:n_samples], u[:n_samples]]).T
    params = np.linalg.lstsq(A, y_true[1:], rcond=None)[0]
    a1, b0 = params[0], params[1]
    print(f"Fitted ARMA Parameters: a1 = {a1}, b0 = {b0}")

    # 使用 ARMA 模型进行仿真
    y_arma = np.zeros(n_samples + 1)
    for k in range(1, n_samples):
        y_arma[k] = -a1 * y_arma[k-1] + b0 * u[k]
    return y_arma
    



# 系统参数
dt = 0.01  # Step size
T = 20    # Simulation duration
time = np.arange(0, T , dt)
n_samples = time.size

# 输入初始化
#u(t) = 0.5 sin( 0.6πt)
u1 = 0.5 * np.sin(0.6 * np.pi * np.arange(0, T, dt))
#u(t) = 0.4 cos( 0.4πt) + 0.15 sin( 3πt)
u2 = 0.4 * np.sin(0.4 * np.pi * np.arange(0, T, dt)) + 0.15 * np.sin(3 * np.pi * np.arange(0, T, dt))
y_true = np.zeros(n_samples + 1)

#控制输入的不同获取不同的响应序列
u = u1
sign = '_u1'
y_true = Response_true(y_true,u)


# 准备训练的数据集并训练
# 使用 np.vstack 将 u 和 y_true 垂直堆叠起来，使它们形成一个二维数组，其中每一列分别是 u 和 y_true 的值。
X = np.vstack((u[:n_samples], y_true[:n_samples])).T 
Y = y_true[1:]

# 训练参数
learning_rate = 1e-5
batch_size = 8
max_epochs = 600
FNN_Train(X,Y,learning_rate,batch_size,max_epochs,sign)
    

## 使用神经网络模型进行仿真
# 加载训练好的神经网络模型
model = FNN()  # 定义 FNN 结构
model.load_state_dict(torch.load(f"./Identify_Solution/Neural_Network/fnn_model{sign}.pth"))
model.eval()

# 使用神经网络进行仿真
y_nn = np.zeros(n_samples + 1)
with torch.no_grad():
    for k in range(1, n_samples):
        input_data = torch.tensor([u[k], y_nn[k-1]], dtype=torch.float32).view(1, -1)
        y_nn[k] = model(input_data).item()
        
        
## 使用ARMA模型进行预测
y_arma =  ARMA_Predict(y_true,u)


## 仿真结果对比图
plt.figure(figsize=(12, 6))
plt.plot(time,y_true[:n_samples], label="True Output", color="blue")
plt.plot(time,y_arma[:n_samples], label="ARMA Model Output", color="green", linestyle="--")
plt.plot(time,y_nn[:n_samples], label="FNN Output", color="red", linestyle="--")
plt.xlim(0, T)  # 将横坐标限制在 0 到 T
plt.xlabel("T/s")
plt.ylabel("Output y[k]")
plt.title("System Output: True vs ARMA Model vs FNN")
plt.legend()
plt.legend(loc="upper right") # 图例放在右上角
plt.savefig(f"./Identify_Solution/Neural_Network/FNN_ARMA_Resultscontrast{sign}.png", dpi=300, bbox_inches='tight')


## 跟踪误差图
error_arma = y_true[:n_samples] - y_arma[:n_samples]
error_nn = y_true[:n_samples] - y_nn[:n_samples]
plt.figure(figsize=(12, 6))
plt.plot(time,error_arma, label="ARMA Model Error", color="green")
plt.plot(time,error_nn, label="FNN Error", color="red")
plt.xlim(0, T)  # 将横坐标限制在 0 到 T
plt.xlabel("T/s")
plt.ylabel("Tracking Error")
plt.title("Tracking Error: ARMA Model vs FNN")
plt.legend()
plt.legend(loc="upper right") # 图例放在右上角
plt.savefig(f"./Identify_Solution/Neural_Network/FNN_ARMA_TrackError{sign}.png", dpi=300, bbox_inches='tight')


plt.show()


