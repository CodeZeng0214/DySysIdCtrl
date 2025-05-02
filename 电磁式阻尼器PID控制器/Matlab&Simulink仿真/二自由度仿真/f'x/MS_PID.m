clear;clc

%% 系统参数
m = 1.6; % 电磁吸振器器质量
M = 100; % 待减振对象质量
k_m = 3000; % 电磁吸振器刚度
k_M = 200000; % 平台刚度
k_f = 45; % 电—力常数 N/A 来自要求文件
k_E = 0; % 作动器反电动势系数 % 来自图片参数
L = 0.0045; % 线圈的电感 来自图片参数
R_m = 5; % 线圈的电阻 % 来自图片参数
c_m = 0.2; % 电磁吸振器阻尼
c_M = 1; % 平台阻尼
is_a = true; % 是否加速度反馈
w_m = sqrt(k_m/m); % 吸振器固有频率
w_M = sqrt(k_M/M); % 被控物体固有频率

num = 1;
den = [L R_m];

%% 有控制的状态空间方程建立（位移和速度为状态量，输入取地基的位移）
Ac = [0      1       0           0;
    -k_m/m -c_m/m   k_m/m       c_m/m;
     0      0       0           1;
     k_m/M  c_m/M -(k_m+k_M)/M -(c_m+c_M)/M];
Bc = [0 +k_f/m 0 -k_f/M]';
Cc = [-k_m/m -c_m/m   k_m/m       c_m/m;
      k_m/M  c_m/M -(k_m+k_M)/M -(c_m+c_M)/M];
Dc = [+k_f/m -k_f/M]';
E_X = eye(4);
Ec = [0 0 0 c_M/M;
     0 0 0 k_M/M]';
system = ss(Ac,Bc,Cc,Dc);
tf_system = tf(system);
x_system = tf_system(2,1);
bode(x_system)

%% 无控制的状态空间建立
An = [0 1;
    -k_M/M -c_M/M];
Bn = [0 0]';
Cn = [1 0];
Dn = 0;
En = [0 0 ;
    c_M/M k_M/M];

%% PID控制器
amp = 0.0005; % 振幅
fre = 45; % 频率
% PID参数
if is_a
    % 加速度反馈
    Kp = 1;    % 比例系数
    Ki = 0;     % 积分系数
    Kd = 0;     % 微分系数
else
    % 位移反馈
    Kp = 27000;    % 比例系数
    Ki = 13000;     % 积分系数
    Kd = 00070;     % 微分系数
end

sim('Model2020a.slx');
