%% 论文《基于pid控制算法的主动式动力吸振器》复现
clear;clc

%% 系统参数
m = 1.6; % 电磁吸振器器质量
M = 14.68; % 待减振对象质量
k_m = 32700; % 电磁吸振器刚度
k_M = 300e3; % 平台刚度
k_f = 45; % 电—力常数 N/A 来自要求文件
k_E = 30; % 作动器反电动势系数 % 来自图片参数
L = 0.0045; % 线圈的电感 来自图片参数
R_m = 5; % 线圈的电阻 % 来自图片参数
c_m = 7; % 电磁吸振器阻尼
c_M = 1; % 平台阻尼
w_m = sqrt(k_m/m)/(2*pi); % 吸振器固有频率
w_M = sqrt(k_M/M)/(2*pi); % 被控物体固有频率
d_M = c_M / (2*sqrt(k_M*M)); % 被控物体阻尼比
d_m = c_m / (2*sqrt(k_m*m)); % 吸振器阻尼比

%% 求解扰动力到被控对象位移的传递函数、控制力到被控对象位移的传递函数、电磁吸振器的传递函数、PID控制器的传递函数
num_F_x = [m c_m k_m]; % 分子系数
den_F_x = [m*M, (c_M*m + c_m*M + c_m*m), (k_M*m + k_m*M + k_m*m + c_m*c_M), (c_M*k_m + c_m*k_M), k_m*k_M]; % 分母系数
num_fc_x = [m 0 0]; % 分子系数
den_fc_x = [m*M, (c_M*m + c_m*M + c_m*m), (k_M*m + k_m*M + k_m*m + c_m*c_M), (c_M*k_m + c_m*k_M), k_m*k_M]; % 分母系数
F_x_tf = tf(num_F_x,den_F_x); % 扰动力到被控对象位移的传递函数
fc_x_tf = tf(num_fc_x,den_fc_x); % 控制力到被控对象位移的传递函数
ea_tf = tf(k_E,[L R_m]); % 电磁作动器传递函数
Kp = 5000;
Ki = 10000;
Kd = 1;
pid_tf = pid(Kp,Ki,Kd); % PID控制器传递函数

%% 求解不同控制方式的系统传递函数
no_control_tf = tf(w_M^2, [1, 2*d_M*w_M, w_M^2]) % 无控制的系统传递函数
passive_control_tf = F_x_tf % 被动控制的系统传递函数
pid_control_tf = F_x_tf / (1 + pid_tf*ea_tf*fc_x_tf) % 加入PID控制器后的系统传递函数


%% 绘图分析
figure; % 新建图形窗口
bode(no_control_tf); % 绘制无控制的系统频率响应图
hold on; % 保持当前图形
bode(passive_control_tf); % 绘制被动控制的系统频率响应图
hold on; % 保持当前图形
bode(pid_control_tf); % 绘制加入PID控制器后的系统频率响应图
legend('无控制系统','被动控制系统','PID控制器'); % 图例
title('频率响应图'); % 标题

%% 仿真分析
t = 0:0.001:10; % 时间范围
r = 0.1*sin(30*t); % 扰动力
un = lsim(no_control_tf,r,t); % 计算无控制系统响应
up = lsim(passive_control_tf,r,t); % 计算被动控制系统响应
uc = lsim(pid_control_tf,r,t); % 计算系统响应
figure; % 新建图形窗口
subplot(2,1,1); % 创建2行1列的子图，当前为第1个子图
plot(t,un,'r'); % 绘制无控制系统响应
xlabel('时间/s'); % x轴标签
legend('无控制系统'); % 图例
subplot(2,1,2); % 当前为第2个子图
plot(t,up,'g'); % 绘制被动控制系统响应
xlabel('时间/s'); % x轴标签
% 继续在第二个子图位置绘制PID控制器响应
hold on; % 保持当前图形
subplot(2,1,2); % 当前为第2个子图
plot(t,uc,'b'); % 绘制加入PID控制器后的系统响应
xlabel('时间/s'); % x轴标签
legend('被动控制系统','PID控制器'); % 图例
title('系统响应'); % 标题
