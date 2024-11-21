% 输出响应序列或者loss
function output = Function_NTI_ResponseLoss(num,den,Td,un,y,timen)

    % 构建传递函数
    G = tf(num, den);
    s = tf('s');
    G_delay =  exp(-Td * s);
    % 合并传递函数和时滞
    G_with_delay = G * G_delay;
    % 使用 Pade 逼近近似时滞
    %delayOrder = 3;  % Pade 逼近的阶数（可以根据精度需求调整）
    %G_with_delay_est = pade(G_with_delay,delayOrder);
    [yn_true, ~] = lsim(G_with_delay, un, timen);

    % 如果有第5个输入参数，计算loss
    if y ~= 1
        loss = 0.5 * sum((yn_true - y).^2);
        output = loss;
    else
        output = yn_true;
    end
end



