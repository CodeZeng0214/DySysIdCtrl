function R = Function_PSO_PID_Rewardfx(Kp,Ki,Kd,sys,dt,T)

    time = 0 : dt : T;

    % 离散化控制系统
    sys_d = c2d(sys,dt);
    [A_d,B_d,C_d,D_d] = ssdata(sys_d);

    % 初始化系统变量矩阵
    x = zeros(length(time), 2);  % 状态变量矩阵，存储每个时刻的状态变量
    y = zeros(length(time), 1);  % 输出矩阵，存储每个时刻的输出
    u = zeros(length(time), 1);  % 系统输入矩阵，存储每个时刻的PID控制器的输出
    e = zeros(length(time), 1);  % 存储每个时刻的系统的误差
    r = zeros(size(time)); % 初始化奖励函数
    x(1,:) = [0 1]; % 初速度响应条件

    for i = 2 : length(time)
        
        % 更新前一时刻值
        if i==2 
            e_prev2 = 0;
        else 
            e_prev2 = e_prev;
        end
        e_prev = e(i-1);
        e(i) = 0 - y(i-1,:);
        
        % 计算增量
        delta_u = Kp * (e(i) - e_prev) + Ki * e(i) * dt + Kd * (e(i) - 2 * e_prev + e_prev2) / dt;
        % 更新控制量
        u(i) = u(i-1) + delta_u;

        % PID 控制器
        %u(i-1) = Kp*e(i-1) + Ki*sum(e) * dt + Kd*(e(i-1) - e_prev)/dt;
        

        % 更新状态
        x(i,:) = (A_d * x(i-1,:)' + B_d * u(i));
        y(i,:) = C_d * x(i,:)' + D_d * u(i);
        
        % 奖励函数设置    
        r(i) = -10*x(i,1)^2 - x(i,2)^2 - 0.1*u(i)^2;
        
    end
    
    % 奖励
    R = sum(r);

    
end

