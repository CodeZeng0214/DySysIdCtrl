function new_obj = discretize_system(obj, Ts)
    % 离散化连续状态空间系统（零阶保持假设）
    % 输入：obj - 系统对象（需包含A、B、E、Ts属性）
    %      Ts - 采样周期（可选，若未传入则使用obj.Ts）
    
    % 处理采样周期（若未传入则使用类的默认值）
    if nargin < 2
        Ts = obj.Ts;  % 假设obj.Ts已预先定义
    else
        obj.Ts = Ts;  % 更新类的采样周期
    end
    
    % 获取矩阵维度
    n = size(obj.A, 1);       % 状态维度（A为n×n）
    m1 = size(obj.B, 2);      % 控制输入维度（B为n×m1）
    m2 = size(obj.E, 2);      % 外部干扰维度（E为n×m2）
    
    % 构造扩展矩阵 M（维度：(n+m1+m2) × (n+m1+m2)）
    M = zeros(n + m1 + m2);
    M(1:n, 1:n) = obj.A;          % 左上角：A矩阵
    M(1:n, n+1:n+m1) = obj.B;     % 中间块：B矩阵
    M(1:n, n+m1+1:end) = obj.E;   % 右侧块：E矩阵
    
    % 计算矩阵指数 exp(M * Ts)
    expM = expm(M * Ts);
    
    % 提取离散化矩阵
    new_obj.Ad = expM(1:n, 1:n);               % 离散状态转移矩阵 Ad
    new_obj.Bd = expM(1:n, n+1:n+m1);          % 离散控制输入矩阵 Bd
    new_obj.Ed = expM(1:n, n+m1+1:end);        % 离散外部干扰矩阵 Ed
end

