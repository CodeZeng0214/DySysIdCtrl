% 输出响应序列或者loss
function output = Function_LTI_ResponseLoss(m,c,k,u,y_true,dt,T)
    time = 0 : dt : T; % 仿真时间向量
    A = [0, 1; -k/m, -c/m];
    B = [0;1/m];
    C = [1 0];
    D = 0;
    sys = ss(A,B,C,D);
    sys_d = c2d(sys,dt);
    [A_d, B_d, C_d, D_d] = ssdata(sys_d);
    x = zeros(length(time), 2);  % 状态变量矩阵，存储每个时刻的状态变量
    y = zeros(length(time), 1);  % 输出矩阵，存储每个时刻的输出
    x(1,:) = 0;
    % 仿真离散系统的输出
    for i = 2 : length(time)
        % 更新状态
        x(i,:) = (A_d * x(i-1,:)' + B_d * u(i-1));
        y(i,:) = C_d * x(i,:)' + D_d * u(i-1);
    end

    % 如果有第5个输入参数，计算loss
    if y_true ~= 1
        loss = 0.5 * sum((y_true - y).^2);
        output = loss;
    else
        output = y;
    end
end



