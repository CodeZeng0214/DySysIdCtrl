clear;
clc;
M = 0.5;%小车质量
m = 0.2;%杆质量
b = 0.1;%小车摩擦系数
l = 0.3;%杆长度
I = 0.006;%杆转动惯量
g = 9.81;

theta_0 = 150*pi/180;
dF = 1;%干扰力矩

AKP = 0.25;
AKI = 0.12;
AKD = 0.05;

PKP = 1;
PKI = 0.0;
PKD = 1;

sim("daolibai_2016b.slx");

figure(1);
set(figure(1), 'Position', [100 100 1000 500]);
subplot(2,1,1);
plot(X,'r','linewidth',2);hold on;
ylim([0,8]);
ylabel('X [m]');
get(figure(1)); set(gca,'FontSize',14); grid on;

subplot(2,1,2);
plot(Angle,'r','linewidth',2);hold on;
ylim([150,195]);
xlabel('t [s]'); ylabel('Angle [°]');
get(figure(1)); set(gca,'FontSize',14); grid on;

