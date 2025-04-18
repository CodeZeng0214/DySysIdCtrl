clear;clc

%% 系统参数
m = 3; % 待隔振载荷 Kg ？
M = 1.34; % 电磁吸振器的总质量 Kg 来自图片参数
k_f = 45; % 来自要求文件
          % 电磁吸振器的电—力常数 N/A
          % 线圈式均匀磁场　km = F/I = NBL  N导线圈数；Ｌ应为缠绕的半径？；Ｂ为磁场强度
L = 0.0045; % 线圈的电感　来自图片参数
c = 0.1; % 电磁吸振器的阻尼 ？
k = 3400; % 系统主弹簧刚度 ？
k_E = 30; % 作动器反电动势系数 来自图片参数
R = 5; % 线圈及导线的总电阻 来自图片参数
is_a = true; % 是否加速度反馈

%% 状态空间方程建立

% A = [0      1       0;
%     -k/m   -c/m     0;
%     -(k_E*k)/m  -(k_E*c)/m  0];
% B = [0      0   0;
%     k_E/m   0   0;
%     (k_E*k_M)/m R L];
% C = [-k/m -c/m k_M/m];
% D = 0;
% E = [0      0   0;
%      k/m    c/m 0;
%      k_E*k/m  k_E*c/m -k_E];
% system = ss(A,B,C,D);

A = [0 1;-k/m -c/m];
B = [0;k_f/m];
C = [1 0];
D = 0;
E = [0 0;c/m k/m];

%% PID控制器
amp = 0.001; % 振幅
fre = 37
% PID参数
if is_a
    % 加速度反馈
    Kp = 0;    % 比例系数
    Ki = 0;     % 积分系数
    Kd = 0;     % 微分系数
else
    % 位移反馈
    Kp = 27000;    % 比例系数
    Ki = 13000;     % 积分系数
    Kd = 00070;     % 微分系数
end

sim('SS_PID_Control_model.slx');