clear;clc

%% 系统参数
m = 1.6; % 电磁吸振器器质量
M = 10; % 待减振对象质量
k_m = 34000; % 电磁吸振器刚度
k_M = 100000; % 平台刚度
k_f = 45; % 电—力常数 N/A 来自要求文件
k_E = 30; % 作动器反电动势系数 % 来自图片参数
L = 0.0045; % 线圈的电感 来自图片参数
R_m = 5; % 线圈的电阻 % 来自图片参数
c_m = 0.1; % 电磁吸振器阻尼
c_M = 1; % 平台阻尼
is_a = true; % 是否加速度反馈

%% 状态空间方程建立（位移和速度为状态量，输入取地基的位移）
A = [0      1       0           0;
    -k_m/m -c_m/m   k_m/m       c_m/m;
     0      0       0           1;
     k_m/M  c_m/M -(k_m+k_M)/M -(c_m+c_M)/M];
B = [0 k_f/m 0 -k_f/M]';
%B = [0 k_m/m1 0 -k_m/m2]';
C = [1 0 0 0;
     0 0 1 0];
D = [0 0]';
E_X = eye(4);
E_Z = [0 0 0 c_M/M;
     0 0 0 k_M/M]';
system = ss(A,B,C,D)
tf_system = tf(system)
x_system = tf_system(1,1)
bode(x_system)
num = 1;
den = [L R_m];
G_system = tf(num,den);

%% PID
amp = 0.01; % 振幅
% PID参数
if is_a
    % 加速度反馈
    Kp = 100000;    % 比例系数
    Ki = 0;     % 积分系数
    Kd = 0;     % 微分系数
else
    % 位移反馈
    Kp = 27000;    % 比例系数
    Ki = 13000;     % 积分系数
    Kd = 00070;     % 微分系数
end