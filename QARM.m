clc; clear all; close all;

export_tables = false;
save_figures = false; % If true, the folder Graphs must exists in the same directory

%------------------------------------- *Quantitative Asset and Risk Management, Project 1* ----------------------------------------%
%----------------------------------------------------------------------------------------------------------------------------------%
%-----------------------------  A liability-relative drawdown approach to pension asset liability management    -------------------%
%----------------------------------------------------------------------------------------------------------------------------------%
%----------------------------------------------------------------------------------------------------------------------------------%
%---   HEC Lausanne, MScF                                                                      ------------------------------------%
%---   Quantitative Asset and Risk Managemen, Pr. Erice Jondeau                                 -----------------------------------%
%---   AUTHORS:                                                                                ------------------------------------%
%---   Romain Pauli (09412099) | Ludovic Mojonnet (09413840) | Guillaume Nagy (09417304)       ------------------------------------%
%----------------------------------------------------------------------------------------------------------------------------------%
%----------------------------------------------------------------------------------------------------------------------------------%

data=xlsread('QARM_DATA.xlsx', 1);  %Import the data from spreadsheet (log returns)

N = length(data(1,:));  %Number of columns of our matrix
T = 120;    % Number of period we want to estimate (quarters)
T_data = length(data(:,1)); %Number of period of our matrix
N_assets = 6;   % Number of asset classes

disp('The excel spreadsheet for 6 Asset Classes has been read and allocated in the matrix data');

%% Simulating data using a Vector Autoregressive Model

%-------------------------------------  Simulating data using a Vector Autoregressive Model  --------------------------------------%
% In this first section, we use a VAR(1) model to simulate expected returns based on the historical observed data. We first
% perform many regressions to estimate the value of each parameters of the model. Thereafter we use this model this to
% estimate the expected return of each asset classes, liability and economic factors. This is done using the simulate_data
% function, which is defined in the file simulate_data.m
% 
%----------------------------------------------------------------------------------------------------------------------------------%

Psi_0 = zeros(1,N); % Creation of empty arrays to stock the future values
Psi_1 = zeros(N,N);
residuals = zeros(T_data-1,N);
std_dev_res = zeros(1,N);

for i=1:N
    reg_VAR(i) = ols(data(2:end,i), [ones(size(data(2:end,i))) data(1:end-1,:)]);   % Results of the VAR(1) process
    Psi_0(i) = reg_VAR(i).beta(1);
    Psi_1(i,1:10) = reg_VAR(i).beta(2:11);
    residuals(:,i) = reg_VAR(i).resid;
    std_dev_res(i) = std(residuals(:,i));
end

Psi_1 = Psi_1';

sim_data = simulate_data(data, Psi_0, Psi_1, std_dev_res, T/4, N);  % Simulation of our data on the base of our VAR(1) process using the simulate data method

expected_log_returns = zeros(T/4, N_assets);    % Creation of empty arrays to stock the future values
expected_excess_log_returns = zeros(T/4, N_assets);

for i=1:N_assets
    expected_log_returns(:,i) = sim_data(:,i);  % Computation of the expected log returns for the 6 asset classes
    expected_excess_log_returns(:,i) = sim_data(:,i) - sim_data(:,7);   % Computation of the expected excess log return (assets - liabilities)  for the 6 asset classes
end

mean_expected_log_returns = mean(expected_log_returns); % Computation of the mean, standard deviation, correlation and covariance for the expected (excess) log returns 
std_dev_expected_log_returns = std(expected_log_returns);
corr_expected_log_returns = corr(expected_log_returns);
cov_expected_log_returns = cov(expected_log_returns);

mean_expected_excess_log_returns = mean(expected_excess_log_returns);
std_dev_expected_excess_log_returns = std(expected_excess_log_returns);
corr_expected_excess_log_returns = corr(expected_excess_log_returns);
cov_expected_excess_log_returns = cov(expected_excess_log_returns);

% Max Drawdown for each asset classes
 
MaxDD_asset_classes = zeros(1, N_assets);   % Creation of empty arrays to stock the future values

for i=1:N_assets    % Computation of the maxdrawdown for the 6 asset classes using the function maxdrawdown_logr defined in maxdrawdown_logr.m
    MaxDD_asset_classes(:,i) = maxdrawdown_logr(expected_excess_log_returns(:,i));  
end

disp('End of section VAR');

%% Mean Variance Frontiers limits

%---------------------------------------------- Mean Variance Frontiers limits ----------------------------------------------------%
% In order to have a comparable basis, we decided to restraint the different efficient frontiers. Indeed, we first compute the
% minimum and maximum reachable excess returns with both asset only and liability relative mean-variance optimizations.
% Thereafter we define the 100 target returns for which we want to minimize the variance and maximum drawdown. The target
% returns are the same for each kind of optimization.
%----------------------------------------------------------------------------------------------------------------------------------%

frontLimitsLR = Portfolio;  %Computation of the minimum and maximum returns reachable with the mean-variance liability relative approach
frontLimitsLR = frontLimitsLR.setAssetMoments(mean_expected_excess_log_returns, cov_expected_excess_log_returns);
frontLimitsLR = frontLimitsLR.setDefaultConstraints;
frontLimitsLR = frontLimitsLR.estimatePortReturn(frontLimitsLR.estimateFrontierLimits);

frontLimitsAO = Portfolio;  %Computation of the minimum and maximum returns reachable with the mean-variance asset only approach
frontLimitsAO = frontLimitsAO.setAssetMoments(mean_expected_log_returns, cov_expected_log_returns);
frontLimitsAO = frontLimitsAO.setDefaultConstraints;
frontLimitsAO = frontLimitsAO.estimatePortReturn(frontLimitsAO.estimateFrontierLimits);
frontLimitsAO = frontLimitsAO - mean(sim_data(:,7));

minFron = max(frontLimitsLR(1), frontLimitsAO(1))+0.001; % We use the maximum of the minimums as lower limit
maxFron = min(frontLimitsLR(2), frontLimitsAO(2))-0.001; % We use the minimum of the maximums as upper limit

fronStep = (maxFron-minFron)/100;   % The range between each target returns

efficient_frontier_target_excess_returns = [minFron:fronStep:maxFron];  % Create a vector with all the target returns
efficient_frontier_nb_port = round(length(efficient_frontier_target_excess_returns));   % Compute the number of target returns (100)
ex_number = round([efficient_frontier_nb_port/4 2*(efficient_frontier_nb_port/4) 3*(efficient_frontier_nb_port/4)]);    % 25th 50th and 75th target return

disp('End of section Mean Variance Frontiers limits');


%% Mean Variance Efficient Frontier Liability Relative

%------------------------------------ Mean Variance Efficient Frontier Liability Relative  ----------------------------------------%
% We computed the efficient frontier for the mean variance approach relative to the liabilities. To do so, we have use the
% portopt function. In order to simplify our model, we took into account the log returns only (the discussion of computed it
% correctly will be explained in the limit of our model).
%----------------------------------------------------------------------------------------------------------------------------------%

%Return the standard deviation, expected excess return and weights of the portfolios on the efficient frontier using the
% portopt function
[PortRisk_LR, Port_excess_Return_LR, PortWts_LR] = portopt(mean_expected_excess_log_returns, cov_expected_excess_log_returns, [], efficient_frontier_target_excess_returns);

MaxDD_MV_LR = zeros(efficient_frontier_nb_port, 1); % Creation of empty arrays to stock the future values
Period_MaxDD_MV_LR = zeros(efficient_frontier_nb_port, 2);

for i=1:length(PortWts_LR)  % Compute the maximum drawdown and the period when it occurs for each optimal portfolio
    [MaxDD_MV_LR(i,1), Period_MaxDD_MV_LR(i,1:2)] = maxdrawdown_logr((PortWts_LR(i,:)*expected_excess_log_returns')');
end

ex_MV_LR_Portfolio = zeros(length(ex_number), N_assets); % Creation of empty arrays to stock the future values
ex_MV_LR_Portfolio_returns = zeros(length(ex_number), T/4);

for k=1:length(ex_number)   % Select three portfolio (25th, 50th, 75th target returns)
    ex_MV_LR_Portfolio(k,:) = PortWts_LR(ex_number(k),:);
    ex_MV_LR_Portfolio_returns(k,:) = ex_MV_LR_Portfolio(k,:) * expected_excess_log_returns';
end

disp('End of section Mean Variance Efficient Frontier Liability Relative');

%% Mean Variance Efficient Frontier Assets Only

%------------------------------------  Mean Variance Efficient Frontier Assets Only -----------------------------------------------%
% We computed the efficient frontier for the mean variance approach only related to the asset. To do so, we have use the
% portopt function. In order to simplify our model, we took into account the log returns only (the discussion of computed it
% correctly will be explained in the limit of our model).
%----------------------------------------------------------------------------------------------------------------------------------%

Target_excess_Return_AO = Port_excess_Return_LR + mean(sim_data(:,7)); % We took the previous target returns, and added the liabilities, since it is not taken into account while computing the efficient frontier

% Return the standard deviation, expected excess return and weights of the portfolios on the efficient frontier using the portopt function
[PortRisk_AO, PortReturn_AO, PortWts_AO] = portopt(mean_expected_log_returns, cov_expected_log_returns, [], Target_excess_Return_AO);

Port_excess_Return_AO = PortReturn_AO - mean(sim_data(:,7));    % We deduce the mean again, in order to have comparable data
PortRisk_excess_AO = std((PortWts_AO*expected_excess_log_returns')');   % We compute the standard deviation on the base of the expected excess log returns

MaxDD_MV_AO = zeros(efficient_frontier_nb_port, 1); % Creation of empty arrays to stock the future values
Period_MaxDD_MV_AO = zeros(efficient_frontier_nb_port, 2);

for i=1:length(PortWts_AO)   % Compute the maximum drawdown and the period when it occurs for each optimal portfolio
    [MaxDD_MV_AO(i,1), Period_MaxDD_MV_AO(i,1:2)] = maxdrawdown_logr((PortWts_AO(i,:)*(expected_excess_log_returns)')');
end

ex_MV_AO_Portfolio = zeros(length(ex_number), N_assets);   % Creation of empty arrays to stock the future values
ex_MV_AO_Portfolio_returns = zeros(length(ex_number), T/4);

for k=1:length(ex_number)   % Select three portfolio (25th, 50th, 75th target returns)
    ex_MV_AO_Portfolio(k,:) = PortWts_AO(ex_number(k),:);
    ex_MV_AO_Portfolio_returns(k,:) = ex_MV_AO_Portfolio(k,:)*expected_excess_log_returns';
end

disp('End of section MV AO');

%% Max Min DD Optimization Assets Only

%------------------------------------------ Max Min DD Optimization Assets Only ---------------------------------------------------%
% We computed the efficient frontier for the maximum drawdown approach relative to the liabilities. To do so, we have use the
% fmincon function to minimize the maximum drawdown computed in the function maxdrawdown_logr.m In order to simplify our model, 
% we took into account the log returns only (the discussion of computed it correctly will be explained in the limit of our model).
%----------------------------------------------------------------------------------------------------------------------------------%

target_returns_AO = PortReturn_AO;  % We took the previous target returns

w_AO = zeros(efficient_frontier_nb_port, N_assets); % Creation of empty arrays to stock the future values
MaxDD_DD_AO = zeros(efficient_frontier_nb_port, 1);

% Optimisation with the fmincon
for i=1:length(target_returns_AO)   % For each target return, we use fmincon to minimize the maximum drawdown
    w0 = [1 0 0 0 0 0]; % Initial value, will change with the optimization
    Aeq = [ones(1,6); mean_expected_log_returns];   % Constraints: Aeq*w==beq
    beq = [ones(1,1) target_returns_AO(i)];         
    A = (-eye(6));  % Constraints: A*w=<b
    b = zeros(1,6); 
    options  =  optimset('fmincon'); % Get the default option for fmincon
    options = optimset(options, 'MaxFunEvals',100000, 'Algorithm', 'active-set', 'MaxIter', 50000,'Display', 'off'); % Update fmincon options
    [w_AO(i,:),MaxDD_DD_AO(i)] = fmincon(@min_function_AO,w0,A,b,Aeq,beq, [] , [] , [] , options);  %Optimization function, returns the optimal weights and maximum drawdown
    if i==25 | i ==50 | i==75 | i==100
        disp(['DD AO (' num2str(i) '%)']); % Print a message in the console with the status of the optimization
    end
end

std_port_DD_AO = zeros(1, efficient_frontier_nb_port);  % Creation of empty arrays to stock the future values
MaxDD_DD_AO_excess = zeros(efficient_frontier_nb_port, 1);
Period_MaxDD_DD_AO = zeros(efficient_frontier_nb_port, 2);

for i=1:length(w_AO)    % Computation of the standard deviation for the maxdrawdown
    std_port_DD_AO(i) = std((w_AO(i,:)*expected_excess_log_returns')');
    [MaxDD_DD_AO_excess(i,1), Period_MaxDD_DD_AO(i,1:2)] = maxdrawdown_logr((w_AO(i,:)*expected_excess_log_returns')');
end

ex_DD_AO_Portfolio = zeros(length(ex_number), N_assets);    % Computation of the standard deviation for the maxdrawdown
ex_DD_AO_Portfolio_returns = zeros(length(ex_number), T/4);

for k=1:length(ex_number)   % Select three portfolio (25th, 50th, 75th target returns)
    ex_DD_AO_Portfolio(k,:) = w_AO(ex_number(k),:);
    ex_DD_AO_Portfolio_returns(k,:) = ex_DD_AO_Portfolio(k,:)*expected_excess_log_returns';
end

disp('End of section Max Min DD Optimization Assets Only');

%% Max Min DD Optimization Liability Relative

%-------------------------------------------- Max Min DD Optimization Liability Relative ------------------------------------------%
% We computed the efficient frontier for the maximum drawdown approach only related to the assets. To do so, we have use the
% fmincon function to minimize the maximum drawdown computed in the function maxdrawdown_logr.m In order to simplify our model, 
% we took into account the log returns only (the discussion of computed it correctly will be explained in the limit of our
% model).
%----------------------------------------------------------------------------------------------------------------------------------%

target_returns_LR = efficient_frontier_target_excess_returns;   % We took the previous target returns

w_LR = zeros(efficient_frontier_nb_port, N_assets); % Computation of the standard deviation for the maxdrawdown
MaxDD_DD_LR = zeros(efficient_frontier_nb_port, 1);

% Optimisation with the fmincon
for i=1:length(target_returns_LR)    % For each target return, we use fmincon to minimize the maximum drawdown
    w0 = [1 0 0 0 0 0]; % Initial value, will change with the optimization
    Aeq = [ones(1,6); mean_expected_excess_log_returns];    % Constraints: Aeq*w==beq
    beq = [ones(1,1) target_returns_LR(i)]; 
    A = (-eye(6));  % Constraints: A*w=<b
    b = zeros(1,6);
    options  =  optimset('fmincon');    % Get the default option for fmincon
    options = optimset(options, 'MaxFunEvals',100000, 'Algorithm', 'active-set', 'MaxIter', 50000,'Display', 'off');    % Update fmincon options
    [w_LR(i,:),MaxDD_DD_LR(i)] = fmincon(@min_function_LR,w0,A,b,Aeq,beq, [] , [] , [] , options);  %Optimization function, returns the optimal weights and maximum drawdown
    if i==25 | i ==50 | i==75 | i==100
        disp(['DD LR (' num2str(i) '%)']);  % Print a message in the console with the status of the optimization
    end
end

std_port_DD_LR = zeros(1, efficient_frontier_nb_port);  % Creation of empty arrays to stock the future values
MaxDD_DD_LR_2 = zeros(efficient_frontier_nb_port, 1);
Period_MaxDD_DD_LR = zeros(efficient_frontier_nb_port, 2);

for i=1:length(w_LR)     % Computation of the standard deviation for the maxdrawdown
    std_port_DD_LR(i) = std((w_LR(i,:)*expected_excess_log_returns')');
    [MaxDD_DD_LR_2(i,1), Period_MaxDD_DD_LR(i,1:2)] = maxdrawdown_logr((w_LR(i,:)*expected_excess_log_returns')');
end

ex_DD_LR_Portfolio = zeros(length(ex_number), N_assets);     % Computation of the standard deviation for the maxdrawdown
ex_DD_LR_Portfolio_returns = zeros(length(ex_number), T/4);

for k=1:length(ex_number)
    ex_DD_LR_Portfolio(k,:) = w_LR(ex_number(k),:); % Select three portfolio (25th, 50th, 75th target returns)
    ex_DD_LR_Portfolio_returns(k,:) = ex_DD_LR_Portfolio(k,:)*expected_excess_log_returns';
end

disp('End of section Max Min DD Optimization Liability Relative');

%% Figures

%---------------------------------------------------       Figures      -----------------------------------------------------------%
% Here are the different plots related to the 4 previous approaches.
%----------------------------------------------------------------------------------------------------------------------------------%

EfficientFrontierMV = figure(1);    % Draw the efficient frontiers to compare the returns with the standard deviation
plot(PortRisk_excess_AO, Port_excess_Return_AO, 'color', 'red');
hold on 
plot(PortRisk_LR, Port_excess_Return_LR, 'color', 'green');
hold on 
plot(std_port_DD_AO, Port_excess_Return_AO, 'color', 'magenta');
hold on
plot(std_port_DD_LR, Port_excess_Return_LR, 'color', 'blue');
hold on
scatter(std_dev_expected_excess_log_returns, mean_expected_excess_log_returns);
textLabels = {'Cash', 'Equity','Bonds','Commodities','Real Estate','Hedge Funds'};
dx = 0.004; dy = 0.002; 
text(std_dev_expected_excess_log_returns+dx, mean_expected_excess_log_returns+dy, textLabels);
ylim(get(gca, 'ylim') + [-0.005 0.005]);
xlim(get(gca, 'xlim') + [0 0.05]);
title('Efficient Frontier')
legend('MV AO','MV LR','DD AO','DD LR')
xlabel('Standard deviation')
ylabel('Returns')
if save_figures == true;
    saveas(EfficientFrontierMV,'graphs/1EfficientFrontierMV.eps', 'psc2')
end

EfficientFrontierDD = figure(2);     % Draw the efficient frontiers to compare the returns with the maximum drawdown
plot(MaxDD_MV_AO, Port_excess_Return_AO, 'color', 'red');
hold on
plot(MaxDD_MV_LR, Port_excess_Return_LR, 'color', 'green');
hold on
plot(MaxDD_DD_AO_excess, Port_excess_Return_AO, 'color', 'magenta');
hold on
plot(MaxDD_DD_LR_2, Port_excess_Return_LR, 'color', 'blue');
hold on 
scatter(MaxDD_asset_classes, mean_expected_excess_log_returns);
textLabels = {'Cash', 'Equity','Bonds','Commodities','Real Estate','Hedge Funds'};
dx = 0.004; dy = 0.002; % displacement so the text does not overlay the data points
text(MaxDD_asset_classes+dx, mean_expected_excess_log_returns+dy, textLabels);
ylim(get(gca, 'ylim') + [-0.005 0.005]);
xlim(get(gca, 'xlim') + [0 0.05]);
title('Efficient Frontier')
legend('MV AO','MV LR','DD AO','DD LR')
xlabel('Maximum Drawdown')
ylabel('Returns')
if save_figures == true;
    saveas(EfficientFrontierDD,'graphs/2EfficientFrontierDD.eps', 'psc2');
end

for k=1:length(ex_number)   % Draw the prices (starting from 1) of the different strategies overtime
    PortfolioExample(k) = figure(k+2);
    subplot(4, 4,[1 12])
    plot(ret2tick_log_returns(ex_MV_AO_Portfolio_returns(k,:)'), 'color', 'green');
    hold on
    plot(ret2tick_log_returns(ex_MV_LR_Portfolio_returns(k,:)'), 'color', 'red');
    hold on
    plot(ret2tick_log_returns(ex_DD_AO_Portfolio_returns(k,:)'), 'color', 'magenta');
    hold on
    plot(ret2tick_log_returns(ex_DD_LR_Portfolio_returns(k,:)'), 'color', 'blue');
    legend('MV AO','MV LR','DD AO','DD LR');
    labels = {'Cash','Equity','Bonds', 'Commodities', 'Real Estate', 'Hedge Funds'};
    label1 = {'','','', '', '', ''};
    label2 = {'','','', '', '', ''};
    label3 = {'','','', '', '', ''};
    label4 = {'','','', '', '', ''};
    for i=1:6
        if ex_MV_AO_Portfolio(k,i) > 0.08
            label1(i) = labels(i);
        end
        if ex_MV_LR_Portfolio(k,i) > 0.08
            label2(i) = labels(i);
        end
        if ex_DD_AO_Portfolio(k,i) > 0.08
            label3(i) = labels(i);
        end
        if ex_DD_LR_Portfolio(k,i) > 0.08
            label4(i) = labels(i);
        end
    end
    subplot(4,4,13)
    h = pie(ex_MV_AO_Portfolio(k,:), label1);
    hp = findobj(h, 'Type', 'patch');
    a = 1;
    if ex_MV_AO_Portfolio(k,1) > 0
        set(hp(a), 'FaceColor', 'g');
        a = a+1;
    end
    if ex_MV_AO_Portfolio(k,2) > 0
        set(hp(a), 'FaceColor', 'r');
        a = a+1;
    end
    if ex_MV_AO_Portfolio(k,3) > 0
        set(hp(a), 'FaceColor', 'b');
        a = a+1;
    end
    if ex_MV_AO_Portfolio(k,4) > 0
        set(hp(a), 'FaceColor', 'm');
        a = a+1;
    end
    if ex_MV_AO_Portfolio(k,5) > 0
        set(hp(a), 'FaceColor', 'magenta');
        a = a+1;
    end
    if ex_MV_AO_Portfolio(k,6) > 0
        set(hp(a), 'FaceColor', 'yellow');
        a = a+1;
    end
    title('MV AO', 'fontweight', 'bold')
    xlabel('Time (years)')
    clear a;
    subplot(4,4,14)
    h = pie(ex_MV_LR_Portfolio(k,:), label1);
    hp = findobj(h, 'Type', 'patch');
    a = 1;
    if ex_MV_LR_Portfolio(k,1) > 0
        set(hp(a), 'FaceColor', 'g');
        a = a+1;
    end
    if ex_MV_LR_Portfolio(k,2) > 0
        set(hp(a), 'FaceColor', 'r');
        a = a+1;
    end
    if ex_MV_LR_Portfolio(k,3) > 0
        set(hp(a), 'FaceColor', 'b');
        a = a+1;
    end
    if ex_MV_LR_Portfolio(k,4) > 0
        set(hp(a), 'FaceColor', 'm');
        a = a+1;
    end
    if ex_MV_LR_Portfolio(k,5) > 0
        set(hp(a), 'FaceColor', 'magenta');
        a = a+1;
    end
    if ex_MV_LR_Portfolio(k,6) > 0
        set(hp(a), 'FaceColor', 'yellow');
        a = a+1;
    end
    title('MV LR', 'fontweight', 'bold')
    xlabel('Time (years)')
    clear a;
    subplot(4,4,15)
    h = pie(ex_DD_AO_Portfolio(k,:), label1);
    hp = findobj(h, 'Type', 'patch');
    a = 1;
    if ex_DD_AO_Portfolio(k,1) > 0
        set(hp(a), 'FaceColor', 'g');
        a = a+1;
    end
    if ex_DD_AO_Portfolio(k,2) > 0
        set(hp(a), 'FaceColor', 'r');
        a = a+1;
    end
    if ex_DD_AO_Portfolio(k,3) > 0
        set(hp(a), 'FaceColor', 'b');
        a = a+1;
    end
    if ex_DD_AO_Portfolio(k,4) > 0
        set(hp(a), 'FaceColor', 'm');
        a = a+1;
    end
    if ex_DD_AO_Portfolio(k,5) > 0
        set(hp(a), 'FaceColor', 'magenta');
        a = a+1;
    end
    if ex_DD_AO_Portfolio(k,6) > 0
        set(hp(a), 'FaceColor', 'yellow');
        a = a+1;
    end
    title('DD AO', 'fontweight', 'bold')
    xlabel('Time (years)')
    clear a;
    subplot(4,4,16)
    h = pie(ex_DD_LR_Portfolio(k,:), label1);
    hp = findobj(h, 'Type', 'patch');
    a = 1;
    if ex_DD_LR_Portfolio(k,1) > 0
        set(hp(a), 'FaceColor', 'g');
        a = a+1;
    end
    if ex_DD_LR_Portfolio(k,2) > 0
        set(hp(a), 'FaceColor', 'r');
        a = a+1;
    end
    if ex_DD_LR_Portfolio(k,3) > 0
        set(hp(a), 'FaceColor', 'b');
        a = a+1;
    end
    if ex_DD_LR_Portfolio(k,4) > 0
        set(hp(a), 'FaceColor', 'm');
        a = a+1;
    end
    if ex_DD_LR_Portfolio(k,5) > 0
        set(hp(a), 'FaceColor', 'magenta');
        a = a+1;
    end
    if ex_DD_LR_Portfolio(k,6) > 0
        set(hp(a), 'FaceColor', 'yellow');
        a = a+1;
    end
    title('DD LR', 'fontweight', 'bold')
    xlabel('Time (years)')
    clear a;
    num = num2str(k*25);
    if save_figures == true;
        saveas(PortfolioExample(k),['graphs/' num2str(k+2) 'PortfolioExample' num '.eps'], 'psc2');
    end
end
clear num;

stepp=round((100)/5);
sett=(0:stepp:100);
j=1;
for i=1:length(sett);
    legend_obs_MV_AO{i}=[num2str(round(Port_excess_Return_AO(j)*10000)/100) '%'];
    legend_obs_MV_LR{i}=[num2str(round(Port_excess_Return_LR(j)*10000)/100) '%'];
    legend_obs_DD_AO{i}=[num2str(round(Port_excess_Return_AO(j)*10000)/100) '%'];
    legend_obs_DD_LR{i}=[num2str(round(target_returns_LR(j)*10000)/100) '%'];
    j=j+stepp;
end
clear j;
clear step;

Weights = figure(6);    % Draw the weights of the portfolios over the different target returns
title('Optimal Portfolios Weights')
subplot(2, 2, 1), area(PortWts_AO)
legend('Cash', 'Equity', 'Bonds', 'Commodities', 'Real Estate', 'Hedge Funds')
% ylabel('Weights of the optimal Portfolios MV AO', 'fontsize', 10)
xlabel('Returns on the optimal Portfolios MV AO', 'fontsize', 10)
set(gca,'XTick',sett,'XTickLabel',legend_obs_MV_AO);
ylim([0 1]);
subplot(2, 2, 2)
area(PortWts_LR)
legend('Cash', 'Equity', 'Bonds', 'Commodities', 'Real Estate', 'Hedge Funds')
% ylabel('Weights of the optimal Portfolios MV LR', 'fontsize', 10)
xlabel('Returns on the optimal Portfolios MV LR', 'fontsize', 10)
set(gca,'XTick',sett,'XTickLabel',legend_obs_MV_LR);
ylim([0 1]);
subplot(2, 2, 3)
area(w_AO)
legend('Cash', 'Equity', 'Bonds', 'Commodities', 'Real Estate', 'Hedge Funds')
% ylabel('Weights of the optimal Portfolios DD AO', 'fontsize', 10)
xlabel('Returns on the optimal Portfolios DD AO', 'fontsize', 10)
set(gca,'XTick',sett,'XTickLabel',legend_obs_DD_AO);
ylim([0 1]);
subplot(2, 2, 4)
area(w_LR)
legend('Cash', 'Equity', 'Bonds', 'Commodities', 'Real Estate', 'Hedge Funds')
% ylabel('Weights of the optimal Portfolios DD LR', 'fontsize', 10)
xlabel('Returns on the optimal Portfolios DD LR', 'fontsize', 10)
set(gca,'XTick',sett,'XTickLabel',legend_obs_DD_LR);
ylim([0 1]);
if save_figures == true;
    saveas(Weights,'graphs/6Weights.eps', 'psc2')
end
disp('End of section Figures');


%% Random Weights

%----------------------------------------------------  Random Weights  ------------------------------------------------------------%
% In order to verify our previous results, we generated a bench of portfolios (10000) with random allocation in order to test the
% efficient frontier. 
%----------------------------------------------------------------------------------------------------------------------------------%


clear randomWeights;
clear randomPortReturns;

RN = 10000; % Number of portfolios

randomWeights = zeros(RN, N_assets);    % Creation of empty arrays to stock the future values

for i=1:RN  % Generate random weights using the function defined in generateWeights.m
    randomWeights(i,:) = generateWeights;
end

randomPortReturns = randomWeights*expected_excess_log_returns'; % Compute the returns for the different random weights

%% Plot of the random weights

Random_EfficientFrontierMV = figure(8); % Draw the different portfolios in term of returns and standard deviation
scatter(std(randomPortReturns'), mean(randomPortReturns'), 'MarkerEdgeColor','blue');
hold on 
plot(PortRisk_LR, Port_excess_Return_LR, 'color', 'green', 'LineWidth', 1.3);
title('Efficient Frontier')
legend('Random Allocations','MV LR')
xlabel('Standard deviation')
ylabel('Returns')
if save_figures == true;
    saveas(Random_EfficientFrontierMV,'graphs/8RandomEfficientFrontierMV.eps', 'psc2') 
end

Random_EfficientFrontierDD = figure(9); % Draw the different portfolios in term of returns and maximum drawdown
scatter(maxdrawdown_logr(randomPortReturns'), mean(randomPortReturns'), 'MarkerEdgeColor','magenta');
hold on 
plot(MaxDD_DD_LR_2, Port_excess_Return_LR, 'color', 'blue', 'LineWidth', 1.3);
title('Efficient Frontier')
legend('Random Allocations', 'DD LR')
xlabel('Maximum Drawdown')
ylabel('Returns')
if save_figures == true;
    saveas(Random_EfficientFrontierDD,'graphs/9RandomEfficientFrontierDD.eps', 'psc2')
end
disp('End of section Random Weights');


%% 100 Additional simulations

%--------------------------------------------- 100 Additional simulations  --------------------------------------------------------%
% In order to test the different strategies, we simulate 100 VAR(1) models. Thereafter, we observe three specific portfolios
% (25th, 50th, 75th target returns) to see how they behave in the different situations.
%----------------------------------------------------------------------------------------------------------------------------------%

clear simulations
clear expected_excess_returns_sim2
simulations = zeros(size(sim_data));    % Creation of empty arrays to stock the future values
expected_excess_log_returns_sim2 = zeros(size(expected_excess_log_returns));
NumberSim2 = 100;

for j=1:NumberSim2
    simulations(:,:,j) = simulate_data(data, Psi_0, Psi_1, std_dev_res, T/4, N);    % 100 simulations of the VAR(1) model
    for i=1:6
        expected_excess_log_returns_sim2(:,i,j) = simulations(:,i,j) - simulations(:,7,j);  % Computation of the expected excess log returns associated to the different asset classes and simulations
    end
end

mean_expected_excess_log_returns_sim2_MV_AO = zeros(NumberSim2, length(ex_number)); % Creation of empty arrays to stock the future values
mean_expected_excess_log_returns_sim2_MV_LR = zeros(NumberSim2, length(ex_number));
mean_expected_excess_log_returns_sim2_DD_AO = zeros(NumberSim2, length(ex_number));
mean_expected_excess_log_returns_sim2_DD_LR = zeros(NumberSim2, length(ex_number));
std_expected_excess_log_returns_sim2_MV_AO = zeros(NumberSim2, length(ex_number));
std_expected_excess_log_returns_sim2_MV_LR = zeros(NumberSim2, length(ex_number));
std_expected_excess_log_returns_sim2_DD_AO = zeros(NumberSim2, length(ex_number));
std_expected_excess_log_returns_sim2_DD_LR = zeros(NumberSim2, length(ex_number));
MaxDD_sim2_MV_AO = zeros(NumberSim2, length(ex_number));
MaxDD_sim2_MV_LR = zeros(NumberSim2, length(ex_number));
MaxDD_sim2_DD_AO = zeros(NumberSim2, length(ex_number));
MaxDD_sim2_DD_LR = zeros(NumberSim2, length(ex_number));

% Computation of the mean, standard deviation and maxdrawdown of the expected excess log returns for our 3 porfolios
for k=1:length(ex_number)
    for j=1:100
        mean_expected_excess_log_returns_sim2_MV_AO(j,k) = mean(ex_MV_AO_Portfolio(k,:)*expected_excess_log_returns_sim2(:,:,j)');
        mean_expected_excess_log_returns_sim2_MV_LR(j,k) = mean(ex_MV_LR_Portfolio(k,:)*expected_excess_log_returns_sim2(:,:,j)');
        mean_expected_excess_log_returns_sim2_DD_AO(j,k) = mean(ex_DD_AO_Portfolio(k,:)*expected_excess_log_returns_sim2(:,:,j)');
        mean_expected_excess_log_returns_sim2_DD_LR(j,k) = mean(ex_DD_LR_Portfolio(k,:)*expected_excess_log_returns_sim2(:,:,j)');

        std_expected_excess_log_returns_sim2_MV_AO(j,k) = std(ex_MV_AO_Portfolio(k,:)*expected_excess_log_returns_sim2(:,:,j)');
        std_expected_excess_log_returns_sim2_MV_LR(j,k) = std(ex_MV_LR_Portfolio(k,:)*expected_excess_log_returns_sim2(:,:,j)');
        std_expected_excess_log_returns_sim2_DD_AO(j,k) = std(ex_DD_AO_Portfolio(k,:)*expected_excess_log_returns_sim2(:,:,j)');
        std_expected_excess_log_returns_sim2_DD_LR(j,k) = std(ex_DD_LR_Portfolio(k,:)*expected_excess_log_returns_sim2(:,:,j)');

        MaxDD_sim2_MV_AO(j,k) = maxdrawdown_logr((ex_MV_AO_Portfolio(k,:)*(expected_excess_log_returns_sim2(:,:,j))')');
        MaxDD_sim2_MV_LR(j,k) = maxdrawdown_logr((ex_MV_LR_Portfolio(k,:)*(expected_excess_log_returns_sim2(:,:,j))')');
        MaxDD_sim2_DD_AO(j,k) = maxdrawdown_logr((ex_DD_AO_Portfolio(k,:)*(expected_excess_log_returns_sim2(:,:,j))')');
        MaxDD_sim2_DD_LR(j,k) = maxdrawdown_logr((ex_DD_LR_Portfolio(k,:)*(expected_excess_log_returns_sim2(:,:,j))')');
    end
end

% Plot the histograms of the expected returns, standard deviation and maximum drawdown for our 3 portfolios and 4 strategies
for k=1:length(ex_number)
    histogramsSim2(k) = figure(k+9);
    subplot(4,3,1)
    hist(mean_expected_excess_log_returns_sim2_MV_AO(:,k), -0.04:0.01:0.08);
    title('Returns MV AO');
    ylim([0 50])
    xlim([-0.04 0.08])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','green','EdgeColor','black')
    subplot(4,3,2)
    hist(std_expected_excess_log_returns_sim2_MV_AO(:,k), 0.04:0.01:0.18);
    title('Std MV AO');
    ylim([0 50])
    xlim([0.04 0.18])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','green','EdgeColor','black')
    subplot(4,3,3)
    hist(MaxDD_sim2_MV_AO(:,k), 0:0.05:0.8);
    title('MaxDD MV AO');
    ylim([0 50])
    xlim([0 0.8])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','green','EdgeColor','black')
    subplot(4,3,4)
    hist(mean_expected_excess_log_returns_sim2_MV_LR(:,k), -0.04:0.01:0.08);
    title('Returns MV LR');
    ylim([0 50])
    xlim([-0.04 0.08])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','red','EdgeColor','black')
    subplot(4,3,5)
    hist(std_expected_excess_log_returns_sim2_MV_LR(:,k), 0.04:0.01:0.18);
    title('Std MV LR');
    ylim([0 50])
    xlim([0.04 0.18])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','red','EdgeColor','black')
    subplot(4,3,6)
    hist(MaxDD_sim2_MV_LR(:,k), 0:0.05:0.8);
    title('MaxDD MV LR');
    ylim([0 50])
    xlim([0 0.8])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','red','EdgeColor','black')
    subplot(4,3,7)
    hist(mean_expected_excess_log_returns_sim2_DD_AO(:,k), -0.04:0.01:0.08);
    title('Returns DD AO');
    ylim([0 50])
    xlim([-0.04 0.08])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','magenta','EdgeColor','black')
    subplot(4,3,8)
    hist(std_expected_excess_log_returns_sim2_DD_AO(:,k), 0.04:0.01:0.18);
    title('Std DD AO');
    ylim([0 50])
    xlim([0.04 0.18])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','magenta','EdgeColor','black')
    subplot(4,3,9)
    hist(MaxDD_sim2_DD_AO(:,k), 0:0.05:0.8);
    title('MaxDD DD AO');
    ylim([0 50])
    xlim([0 0.8])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','magenta','EdgeColor','black')
    subplot(4,3,10)
    hist(mean_expected_excess_log_returns_sim2_DD_LR(:,k), -0.04:0.01:0.08);
    title('Returns DD LR');
    ylim([0 50])
    xlim([-0.04 0.08])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','blue','EdgeColor','black')
    subplot(4,3,11)
    hist(std_expected_excess_log_returns_sim2_DD_LR(:,k), 0.04:0.01:0.18);
    title('Std DD LR');
    ylim([0 50])
    xlim([0.04 0.18])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','blue','EdgeColor','black')
    subplot(4,3,12)
    hist(MaxDD_sim2_DD_LR(:,k), 0:0.05:0.8);
    title('MaxDD DD LR');
    ylim([0 50])
    xlim([0 0.8])
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','blue','EdgeColor','black')
    num = num2str(k*25);
    if save_figures == true;
        saveas(histogramsSim2(k),['graphs/' num2str(k+9) 'histogramsSim2' num '.eps'], 'psc2')
    end
end
clear num;

disp('End of section 100 Additional simulations');

%% Max Min DD Optimization Liability Relatives with boundaries

%-------------------------------------- Max Min DD Optimization Liability Relatives with boundaries -------------------------------%
% From this section until the end, we are only interested in the maximum drawdown liability relative approach. In this
% section, we impose constraints on the weights of the different portfolios.
%----------------------------------------------------------------------------------------------------------------------------------%

w_LR_Cs_waitForIt = zeros(efficient_frontier_nb_port, N_assets);    % Creation of empty arrays to stock the future values
MaxDD_DD_LR_Cs = zeros(1, efficient_frontier_nb_port);
w_LR_Cs = zeros(efficient_frontier_nb_port, N_assets);

for i=1:length(target_returns_LR)   % Fmincon, computed as in the previous sections with different constraints
    w0 = [1 0 0 0 0 0];
    Aeq = [ones(1,6); mean_expected_excess_log_returns];
    beq = [ones(1,1) target_returns_LR(i)]; 
    A = [-eye(6); eye(6)];
    b = [0 0 0 0 0 0 0.25 0.25 0.25 0.25 0.25 0.25]';   % Constraints: maximum weights = 25%
    options  =  optimset('fmincon');
    options = optimset(options, 'MaxFunEvals',10000, 'Algorithm', 'active-set', 'MaxIter', 50000, 'display', 'off');
    [w_LR_Cs_waitForIt(i,:),MaxDD_DD_LR_Cs(i), exitflag] = fmincon(@min_function_LR,w0,A,b,Aeq,beq, [] , [] , [] , options);
    if exitflag == -2
    	w_LR_Cs(i,:) = [0 0 0 0 0 0];
    else
    	w_LR_Cs(i,:) = w_LR_Cs_waitForIt(i,:);
    end
    if i==25 | i ==50 | i==75 | i==100
        disp(['DD AO (' num2str(i) '%)']);
    end
end

std_port_DD_LR_Cs = zeros(1, efficient_frontier_nb_port);   % Creation of empty arrays to stock the future values
MaxDD_DD_LR_2_Cs = zeros(efficient_frontier_nb_port, 1);
Period_MaxDD_DD_LR_Cs = zeros(efficient_frontier_nb_port, 2);

for i=1:length(w_LR_Cs) % Computation of the standard deviation for the maxdrawdown
    std_port_DD_LR_Cs(i) = std((w_LR_Cs(i,:)*expected_excess_log_returns')');
    [MaxDD_DD_LR_2_Cs(i,1), Period_MaxDD_DD_LR_Cs(i,1:2)] = maxdrawdown_logr((w_LR_Cs(i,:)*expected_excess_log_returns')');
end

Index_Cs = find(MaxDD_DD_LR_2_Cs);  % Return the index of the first and last non zero value in the vector
ex_number_Cs = ex_number(1:end-1);  % We are not interested in the 75th portfolio anymore (difficult to reach with boundaries)

ex_DD_LR_Portfolio_Cs = zeros(length(ex_number_Cs), N_assets);   % Creation of empty arrays to stock the future values
ex_DD_LR_Portfolio_returns_Cs = zeros(2, T/4);

for k=1:length(ex_number_Cs)
    if MaxDD_DD_LR_2_Cs(ex_number(k)) ~= 0  % Check if the portfolios exist (25th and 50th)
        ex_DD_LR_Portfolio_Cs(k,:) = w_LR_Cs(ex_number(k),:);   % Return the weights of the portfolios
        ex_DD_LR_Portfolio_returns_Cs(k,:) = ex_DD_LR_Portfolio_Cs(k,:)*expected_excess_log_returns';   % Return the returns of the portfolios
    end
end

target_returns_LR_Cs = target_returns_LR;   % Same target returns

for i=1:length(MaxDD_DD_LR_2_Cs)    % The following plots of the weights don't take the value impossible to reach
    if MaxDD_DD_LR_2_Cs(i) == 0
        MaxDD_DD_LR_2_Cs(i) = NaN;
        target_returns_LR_Cs(i) = NaN;
    end
end

mean_expected_excess_returns_sim2_DD_LR_Cs = zeros(NumberSim2, length(ex_number_Cs)); % Creation of empty arrays to stock the future values
MaxDD_sim2_DD_LR_Cs = zeros(NumberSim2, length(ex_number_Cs));  

for k=1:length(ex_number_Cs)    % Compute the expected returns and maximum drawdown of our three example portfolios for the 100 additional simulations
    if exist('ex_DD_LR_Portfolio_Cs', 'var');
    for j=1:100
        mean_expected_excess_returns_sim2_DD_LR_Cs(j,k) = mean(ex_DD_LR_Portfolio_Cs(k,:)*expected_excess_log_returns_sim2(:,:,j)');
        MaxDD_sim2_DD_LR_Cs(j,k) = maxdrawdown_logr((ex_DD_LR_Portfolio_Cs(k,:)*(expected_excess_log_returns_sim2(:,:,j))')');
    end
    end
end

disp('End of section Max Min DD Optimization Liability Relatives with boundaries')

%% Plot of the constrained model

%-----------------------------------------------  Plot of the constrained model----------------------------------------------------%
% This section is dedicated to the plot of the constrained model
%----------------------------------------------------------------------------------------------------------------------------------%

FrontierDDLRConstrained = figure(13);    % Draw the two efficient frontiers for the unconstrained and constrained model
plot(MaxDD_DD_LR_2, Port_excess_Return_LR, 'color', 'blue', 'LineWidth', 1.3);
hold on 
plot(MaxDD_DD_LR_2_Cs, Port_excess_Return_LR, 'color', 'cyan');
title('Efficient Frontier')
legend('DD LR','DD LR Constrained')
xlabel('Maximum Drawdown')
ylabel('Returns')
if save_figures == true;
    saveas(FrontierDDLRConstrained,'graphs/13FrontierDDLRConstrained.eps', 'psc2')
end

clear stepp sett j legend_obs_DD_LR;
stepp=round((length(Index_Cs(1):Index_Cs(end)))/5);
sett=Index_Cs(1):stepp:Index_Cs(end);
j=Index_Cs(1);
for i=1:stepp:length(Index_Cs(1):Index_Cs(end))
    legend_fig14{i}=[num2str(round(target_returns_LR(j)*10000)/100) '%'];
    j=j+stepp;
end
clear j;

WeightsDDLRConstrained = figure(14); % Draw the weights of the portfolios over the different target returns
subplot(1,2,1)
area(w_LR(Index_Cs(1):Index_Cs(end),:))
legend('Cash', 'Equity', 'Bonds', 'Commodities', 'Real Estate', 'Hedge Funds')
xlabel('Returns', 'fontsize', 13)
ylim([0 1]);
set(gca,'XTick',sett,'XTickLabel',legend_fig14);
subplot(1,2,2)
area(w_LR_Cs(Index_Cs(1):Index_Cs(end),:))
legend('Cash', 'Equity', 'Bonds', 'Commodities', 'Real Estate', 'Hedge Funds')
xlabel('Returns', 'fontsize', 13)
ylim([0 1]);
set(gca,'XTick',sett,'XTickLabel',legend_fig14);
if save_figures == true;
    saveas(WeightsDDLRConstrained,'graphs/14WeightsDDLRConstrained.eps', 'psc2')
end
% target_returns_LR(Index_Cs(1):Index_Cs(end))

% Plot the histograms of the expected returns and maximum drawdown for the constrained and standard optimization
for k=1:length(ex_number_Cs) 
    if exist('ex_DD_LR_Portfolio_Cs', 'var');
        histogramSim2Constrained(k) = figure(14+k);
        subplot(2,2,1)
        hist(mean_expected_excess_log_returns_sim2_DD_LR(:,k), -0.04:0.01:0.08);
        title('Returns DD LR');
        ylim([0 40])
        xlim([-0.04 0.08])
        h = findobj(gca,'Type','patch');
        set(h,'FaceColor','blue','EdgeColor','black')

        subplot(2,2,2)
        hist(MaxDD_sim2_DD_LR(:,k), 0:0.05:0.8);
        title('MaxDD DD LR');
        ylim([0 40])
        xlim([0 0.8])
        h = findobj(gca,'Type','patch');
        set(h,'FaceColor','blue','EdgeColor','black')

        subplot(2,2,3)
        hist(mean_expected_excess_returns_sim2_DD_LR_Cs(:,k), -0.04:0.01:0.08);
        title('Returns DD LR Constrained');
        ylim([0 40])
        xlim([-0.04 0.08])
        h = findobj(gca,'Type','patch');
        set(h,'FaceColor','yellow','EdgeColor','black')

        subplot(2,2,4)
        hist(MaxDD_sim2_DD_LR_Cs(:,k), 0:0.05:0.8);
        title('MaxDD DD LR Constrained');
        ylim([0 40])
        xlim([0 0.8])
        h = findobj(gca,'Type','patch');
        set(h,'FaceColor','yellow','EdgeColor','black')
    end
    num = num2str(25*k);
    if save_figures == true;
        saveas(histogramSim2Constrained(k),['graphs/' num2str(14+k) 'histogramSim2Constrained' num '.eps'], 'psc2')
    end
end
clear num;

disp('Simulation 2 max DD with constraints');

disp('End of section Plot of the constrained model')

%% Portfolios rebalancing

%--------------------------------------------------  Portfolios rebalancing -------------------------------------------------------%
% In this section we will see the impact of rebalancing the portfolio overtime. To do so, we have taken the portfolio with
% the 25th target return and compared 3 strategies which are: never rebalancing, rebalancing all the 3 years, rebalancing
% every year
%----------------------------------------------------------------------------------------------------------------------------------%

% Compute the evolution of the weights over time if we never revalance, rebalance each 3 years or yearly 
[wts_DD_LR_overtime wts_DD_LR_overtime_reb3 wts_DD_LR_overtime_reb1] = wtsOverTime(ex_DD_LR_Portfolio, expected_excess_log_returns);

WeigthsOverTime = figure(20);    % Draw the weights of the portfolio overtime
subplot(1,3,1)
area(0:T/4-1, wts_DD_LR_overtime)
ylim([0 1])
xlabel('Time (years)')
ylabel('Allocation to each asset class')
title('Never Rebalanced');
subplot(1,3,2)
area(0:T/4-1, wts_DD_LR_overtime_reb3)
ylim([0 1])
xlabel('Time (years)')
title('Rebalanced every 3 years');
ylabel('Allocation to each asset class')
subplot(1,3,3)
area(0:T/4-1, wts_DD_LR_overtime_reb1)
ylim([0 1])
xlabel('Time (years)')
title('Rebalanced yearly');
ylabel('Allocation to each asset class')
if save_figures == true;
    saveas(WeigthsOverTime,'graphs/20WeigthsOverTime.eps', 'psc2');
end

ex_25_reb3_log_excess_returns = sum(wts_DD_LR_overtime_reb3.*expected_excess_log_returns, 2);   % Compute the log excess returns for each strategies
ex_25_reb1_log_excess_returns = sum(wts_DD_LR_overtime_reb1.*expected_excess_log_returns, 2);
ex_25_log_excess_returns = sum(wts_DD_LR_overtime.*expected_excess_log_returns, 2);

WeigthsOverTime_prices = figure(21);  % Draw the prices (starting from 1) of the different strategies overtime
plot(ret2tick_log_returns(ex_25_log_excess_returns), 'color', 'green');
hold on 
plot(ret2tick_log_returns(ex_25_reb3_log_excess_returns), 'color', 'black');
hold on 
plot(ret2tick_log_returns(ex_25_reb1_log_excess_returns), 'color', 'magenta');
legend('Never Rebalanced', '3 Years', '1 Year')
if save_figures == true;
    saveas(WeigthsOverTime_prices,'graphs/21WeigthsOverTimePrices.eps', 'psc2');
end

for i=1:NumberSim2 % Compute the evolution of the weights over time if we never revalance, rebalance each 3 years or yearly for each additional simulation
    [wts_DD_LR_overtime_sim2(:,:,i) wts_DD_LR_overtime_reb3_sim2(:,:,i) wts_DD_LR_overtime_reb1_sim2(:,:,i)]  =  ...
    wtsOverTime(ex_DD_LR_Portfolio, expected_excess_log_returns_sim2(:,:,i));
    ex_25_reb3_log_excess_returns_sim2(:,i) = sum(wts_DD_LR_overtime_reb3_sim2(:,:,i).*expected_excess_log_returns_sim2(:,:,i), 2); % Compute the log excess returns for each strategies
    ex_25_reb1_log_excess_returns_sim_2(:,i) = sum(wts_DD_LR_overtime_reb1_sim2(:,:,i).*expected_excess_log_returns_sim2(:,:,i), 2);
    ex_25_log_excess_returns_sim2(:,i) = sum(wts_DD_LR_overtime_sim2(:,:,i).*expected_excess_log_returns_sim2(:,:,i), 2);
    ex_25_maxDD_sim2(i,1) = maxdrawdown_logr(sum((wts_DD_LR_overtime_sim2(:,:,i).*(expected_excess_log_returns_sim2(:,:,i))), 2)');% Compute the log maximal drawdown for each strategies
    ex_25_reb3_maxDD_sim2(i,1) = maxdrawdown_logr(sum((wts_DD_LR_overtime_reb3_sim2(:,:,i).*(expected_excess_log_returns_sim2(:,:,i))), 2)');
    ex_25_reb1_maxDD_sim2(i,1) = maxdrawdown_logr(sum((wts_DD_LR_overtime_reb1_sim2(:,:,i).*(expected_excess_log_returns_sim2(:,:,i))), 2)');
end

ex_25_reb3_mean_log_excess_returns_sim2 = mean(ex_25_reb3_log_excess_returns_sim2); % Compute the mean of the log excess returns for each strategies
ex_25_reb1_mean_log_excess_returns_sim2 = mean(ex_25_reb1_log_excess_returns_sim_2);
ex_25_mean_log_excess_returns_sim2 = mean(ex_25_log_excess_returns_sim2);

% Plot the histograms of the expected returns and maximum drawdown for the different strategies of rebalancing
histogramWeightsOverTime = figure(22);
subplot(3,2,1)
hist(ex_25_mean_log_excess_returns_sim2, -0.04:0.01:0.08);
title('Never Rebalanced')
ylim([0 48])
xlim([-0.04 0.08])
h = findobj(gca,'Type','patch');
xlabel('Excess Returns')
set(h,'FaceColor','green','EdgeColor','black')

subplot(3,2,2)
hist(ex_25_maxDD_sim2, 0:0.05:0.8);
title('Never Rebalanced')
ylim([0 48])
xlim([0 0.8])
title('Maximum Drawdown')
h = findobj(gca,'Type','patch');
xlabel('Excess Returns')
set(h,'FaceColor','green','EdgeColor','black')

subplot(3,2,3)
hist(ex_25_reb3_mean_log_excess_returns_sim2, -0.04:0.01:0.08);
title('Rebalanced every 3 years')
ylim([0 48])
xlim([-0.04 0.08])
xlabel('Excess Returns')
h = findobj(gca,'Type','patch');
set(h,'FaceColor','black','EdgeColor','white')

subplot(3,2,4)
hist(ex_25_reb3_maxDD_sim2, 0:0.05:0.8);
title('Rebalanced every 3 years')
ylim([0 48])
xlim([0 0.8])
title('Maximum Drawdown')
h = findobj(gca,'Type','patch');
xlabel('Excess Returns')
set(h,'FaceColor','black','EdgeColor','white')

subplot(3,2,5)
hist(ex_25_reb1_mean_log_excess_returns_sim2, -0.04:0.01:0.08);
title('Rebalanced yearly')
ylim([0 48])
xlim([-0.04 0.08])
xlabel('Excess Returns')
h = findobj(gca,'Type','patch');
set(h,'FaceColor','magenta','EdgeColor','black')

subplot(3,2,6)
hist(ex_25_reb1_maxDD_sim2, 0:0.05:0.8);
title('Rebalanced yearly')
ylim([0 48])
xlim([0 0.8])
title('Maximum Drawdown')
h = findobj(gca,'Type','patch');
xlabel('Excess Returns')
set(h,'FaceColor','magenta','EdgeColor','black')
if save_figures == true;
    saveas(histogramWeightsOverTime,'graphs/22histogramWeightsOverTime.eps', 'psc2');
end

disp('End of section Portfolios rebalancing');

%% Simulation with 10 asset classes

%-----------------------------------------  Simulation with 10 asset classes  -----------------------------------------------------%
% We started again from the begining of the project, but instead of taking only 6 asset classes, we take 10. In this section,
% we will analyze the impact of more asset classes on the weights and the efficient frontier of the maximum drawdown maximum
% liability strategy.
%----------------------------------------------------------------------------------------------------------------------------------%


data_10=xlsread('QARM_DATA_2.xlsx', 1); %Import the data from spreadsheet (log returns)

N_10 = length(data_10(1,:));    %Number of columns of our matrix
T_10 = 120;  % Number of period we want to estimate (quarters)
T_data_10 = length(data_10(:,1));   %Number of period of our matrix
N_assets_10 = 10;   % Number of asset classes

% 
% Psi_0_10 = zeros(1,N_10);   % Creation of empty arrays to stock the future values
% Psi_1_10 = zeros(N_10,N_10);
% residuals_10 = zeros(T_data_10-1,N_10);
% std_dev_res_10 = zeros(1,N_10);

for i=1:N_10
    reg_VAR_10(i) = ols(data_10(2:end,i), [ones(size(data_10(2:end,i))) data_10(1:end-1,:)]);   % Results of the VAR(1) process
    Psi_0_10(i) = reg_VAR_10(i).beta(1);
    Psi_1_10(i,1:14) = reg_VAR_10(i).beta(2:15);
    residuals_10(:,i) = reg_VAR_10(i).resid;
    std_dev_res_10(i) = std(residuals_10(:,i));
end

Psi_1_10 = Psi_1_10';

sim_data_10 = simulate_data(data_10, Psi_0_10, Psi_1_10, std_dev_res_10, T_10/4, N_10); % Simulation of our data on the base of our VAR(1) process using the simulate data method

expected_excess_log_returns_10 = zeros(T_10/4, N_assets_10);    % Creation of empty arrays to stock the future values

for i=1:N_assets_10
    expected_excess_log_returns_10(:,i) = sim_data_10(:,i) - sim_data_10(:,11); % Computation of the expected excess log return (assets - liabilities)  for the 10 asset classes
end

mean_expected_excess_log_returns_10 = mean(expected_excess_log_returns_10); % Computation of the mean of the expected excess log return
 
MaxDD_asset_classes_10 = zeros(1, N_assets_10); % Creation of empty arrays to stock the future values

% Computation of the maxdrawdown for the 10 asset classes using the function maxdrawdown_logr defined in maxdrawdown_logr.m
for i=1:N_assets_10
    MaxDD_asset_classes_10(:,i) = maxdrawdown_logr(expected_excess_log_returns_10(:,i));
end

target_returns_LR_10 = efficient_frontier_target_excess_returns;    % We take the previous target returns in order to have comparable data

w_LR_10 = zeros(efficient_frontier_nb_port, N_assets_10);   % Creation of empty arrays to stock the future values
MaxDD_DD_LR_10 = zeros(efficient_frontier_nb_port, 1);

for i=1:length(target_returns_LR_10)    % Fmincon, computed as in the previous sections, but with more asset classes (10 instead of 6)
    w0 = [1 zeros(1, N_assets_10-1)];
    Aeq = [ones(1,10); mean_expected_excess_log_returns_10];
    beq = [ones(1,1) target_returns_LR_10(i)]; 
    A = (-eye(10));
    b = zeros(1,10);
    options  =  optimset('fmincon');
    options = optimset(options, 'MaxFunEvals',100000, 'Algorithm', 'active-set', 'MaxIter', 50000,'Display', 'off');
    [w_LR_10(i,:),MaxDD_DD_LR_10(i)] = fmincon(@min_function_LR_10,w0,A,b,Aeq,beq, [] , [] , [] , options);
    if i==25 | i ==50 | i==75 | i==100
        disp(['DD LR (' num2str(i) '%)']);
    end
end

MaxDD_DD_LR_2_10 = zeros(efficient_frontier_nb_port, 1);    % Creation of empty arrays to stock the future values
Period_MaxDD_DD_LR_10 = zeros(efficient_frontier_nb_port, 2);

 % Compute the maximum drawdown and the period when it occurs for each optimal portfolio
for i=1:length(w_LR_10)
    [MaxDD_DD_LR_2_10(i,1), Period_MaxDD_DD_LR_10(i,1:2)] = maxdrawdown_logr((w_LR_10(i,:)*expected_excess_log_returns_10')');
end

disp('End of section Simulation with 10 asset classes');

%% Plot of the maximum drawdown liability relative with 10 asset classes

%--------------------------- Plot of the maximum drawdown liability relative with 10 asset classes  -------------------------------%
% This section is dedicated to the plot of the maximum drawdown liability relative with 10 asset classes
%----------------------------------------------------------------------------------------------------------------------------------%

FrontierAC10 = figure(23);   % Draw the efficient frontier of the 6 and 10 asset classes maximum drawdown liability relative
plot(MaxDD_DD_LR_2, Port_excess_Return_LR, 'color', 'blue');
hold on 
plot(MaxDD_DD_LR_2_10, Port_excess_Return_LR, 'color', 'black');
hold on 
scatter(MaxDD_asset_classes_10, mean_expected_excess_log_returns_10);
textLabels = {'Cash US', 'Equity US','Bonds US','Commodities US','Real Estate US','Hedge Funds US','Corporate Bonds US','Corporate Bonds Foreign','Equity Foreign', 'Private Equity US'};
dx = 0.004; dy = 0.002; % displacement so the text does not overlay the data points
text(MaxDD_asset_classes_10+dx, mean_expected_excess_log_returns_10+dy, textLabels);
ylim(get(gca, 'ylim') + [-0.005 0.005]);
xlim(get(gca, 'xlim') + [0 0.05]);
title('Efficient Frontier')
legend('DD LR 6 asset classes','DD LR 10 asset classes')
xlabel('Maximum Drawdown')
ylabel('Returns, 10 Asset classes')
if save_figures == true;
    saveas(FrontierAC10,'graphs/23FrontierAC10.eps', 'psc2');
end


WeightsAC10 = figure(24);    % Draw the weights of the portfolios over the different target returns
subplot(1,2,1)
area(target_returns_LR, w_LR)
legend('Cash', 'Equity', 'Bonds', 'Commodities', 'Real Estate', 'Hedge Funds')
xlabel('Returns', 'fontsize', 11)
ylim([0 1]);
subplot(1,2,2)
area(target_returns_LR_10, w_LR_10)
legend('Cash US', 'Equity US','Bonds US','Commodities US','Real Estate US','Hedge Funds US','Corporate Bonds US','Corporate Bonds Foreign','Equity Foreign', 'Private Equity US')
xlabel('Returns, 10 Asset classes', 'fontsize', 11)
ylim([0 1]);
if save_figures == true;
    saveas(WeightsAC10,'graphs/24WeightsAC10.eps', 'psc2');
end

disp('End of section Plot of the maximum drawdown liability relative with 10 asset classes')

%% Limitations
% 

%--------------------------------------------------  Limitations  ---------------------------------------------------------%
%Since we simplified our model in using  the log returns for the cross sectional computations, we want to see the impact of
%such a choise compared to the precise estimations. To do so, we have taken the random weights generated previously, and
%computed in the right way the cross-sectional mean returns.
%----------------------------------------------------------------------------------------------------------------------------------%

cross_sectional_mean_log_returns = exp(expected_excess_log_returns);    % Computation of the cross sectional mean log returns using the right formula
cross_sectional_mean_log_returns = randomWeights*cross_sectional_mean_log_returns';
cross_sectional_mean_log_returns = log(cross_sectional_mean_log_returns);
mean_cross_sectional_mean_log_returns = mean(cross_sectional_mean_log_returns, 2); % Computation of the mean of the cross sectional mean log returns
for i=1:RN
    std_cross_sectional_mean_log_returns(i) = std(cross_sectional_mean_log_returns(i,:));   % Computation of the standard deviation of the cross sectional mean log returns
end
std_cross_sectional_mean_log_returns = std_cross_sectional_mean_log_returns';
maxDD_cross_sectional_mean_log_returns = maxdrawdown_logr(cross_sectional_mean_log_returns');   % Computation of the max drawdown of the cross sectional mean log returns
maxDD_cross_sectional_mean_log_returns = maxDD_cross_sectional_mean_log_returns';


Random_EfficientFrontierMV = figure(25);    % Draw the different portfolios in term of returns and standard deviation
scatter(std_cross_sectional_mean_log_returns, mean_cross_sectional_mean_log_returns, 'MarkerEdgeColor','blue');
hold on
plot(PortRisk_LR, Port_excess_Return_LR, 'color', 'red', 'LineWidth', 1.5);
title('Efficient Frontier')
legend('Random Allocation', 'DD LR')
xlabel('Standard deviation')
ylabel('Returns')
if save_figures == true;
    saveas(Random_EfficientFrontierMV,'graphs/25RandomEfficientFrontierMVLOG.eps', 'psc2') 
end

Random_EfficientFrontierDD = figure(26);    % Draw the different portfolios in term of returns and maximum drawdown
scatter(maxDD_cross_sectional_mean_log_returns, mean_cross_sectional_mean_log_returns, 'MarkerEdgeColor','blue');
hold on 
plot(MaxDD_DD_LR_2, Port_excess_Return_LR, 'color', 'red', 'LineWidth', 1.5);
title('Efficient Frontier')
legend('Random Allocation', 'DD LR')
xlabel('Maximum Drawdown')
ylabel('Returns')
if save_figures == true;
    saveas(Random_EfficientFrontierDD,'graphs/26RandomEfficientFrontierDDLOG.eps', 'psc2')
end

disp('End of section Limitations');

%% Tables

%--------------------------------------------------  Tables  ----------------------------------------------------------------------%
% This sections gererates tables in Latex to be included in the report
%----------------------------------------------------------------------------------------------------------------------------------%



if export_tables == true % Set to false by default at the top of this file.
    a = 0;
    for k=1:length(ex_number)   %   Compute the mean of the mean, standard deviation and maximum drawdown of the 100 simulations for the 3 portfolios (25th, 50th, 75th)
        tableSimulation(k+a,1) = mean(mean_expected_excess_log_returns_sim2_MV_AO(:,k));
        tableSimulation(k+a,2) = mean(mean_expected_excess_log_returns_sim2_MV_LR(:,k));
        tableSimulation(k+a,3) = mean(mean_expected_excess_log_returns_sim2_DD_AO(:,k));
        tableSimulation(k+a,4) = mean(mean_expected_excess_log_returns_sim2_DD_LR(:,k));
        a = a + 1;
        tableSimulation(k+a,1) = mean(std_expected_excess_log_returns_sim2_MV_AO(:,k));
        tableSimulation(k+a,2) = mean(std_expected_excess_log_returns_sim2_MV_LR(:,k));
        tableSimulation(k+a,3) = mean(std_expected_excess_log_returns_sim2_DD_AO(:,k));
        tableSimulation(k+a,4) = mean(std_expected_excess_log_returns_sim2_DD_LR(:,k));
        a = a + 1;
        tableSimulation(k+a,1) = mean(MaxDD_sim2_MV_AO(:,k));
        tableSimulation(k+a,2) = mean(MaxDD_sim2_MV_LR(:,k));
        tableSimulation(k+a,3) = mean(MaxDD_sim2_DD_AO(:,k));
        tableSimulation(k+a,4) = mean(MaxDD_sim2_DD_LR(:,k));
    end


    for i=1:length(ex_number)   % Table of the maximum drawdown and when it occurs for the 4 different strategies (Mean variance and maximum drawdown, asset only and liability relative)
        table1(i,:) = [MaxDD_MV_AO(ex_number(i)) Period_MaxDD_MV_AO(ex_number(i), :) ...
            MaxDD_MV_LR(ex_number(i)) Period_MaxDD_MV_LR(ex_number(i), :) ...
            MaxDD_DD_AO_excess(ex_number(i)) Period_MaxDD_DD_AO(ex_number(i), :) ...
            MaxDD_DD_LR(ex_number(i)) Period_MaxDD_DD_LR(ex_number(i), :)];
    end

    % for i=1:length(names)
    %     names(i) = strrep(names(i), '&', '\&')
    % end
    % 
    hor_tit = {'', 'MV AO', 'MV LR', 'DD AO', 'DD LR'};
    left_name = {'Average Mean & ', 'Average Std Deviation & ', 'Average Maximum Drawdown & '};
    fid = fopen('SimulationTable25.txt','w');
    for b = 1:length(hor_tit);
        fprintf(fid, '%s', char(hor_tit(b)));
        if b < length(hor_tit)
            fprintf(fid, '%s', ' & ');
        end
    end
    fprintf(fid, '%s', ' \\');
    fprintf(fid, '%s', ' \hline ');
    for b = 1:3
        fprintf(fid, '%s', char(left_name(b)));
        for c = 1:4
            fprintf(fid, '%.4f', tableSimulation(b,c));
            if c < 4
                fprintf(fid, '%s', ' & ');
            end
        end
        fprintf(fid, '%s', ' \\');
    end
    fclose(fid)

    fid = fopen('SimulationTable50.txt','w');
    for b = 1:length(hor_tit);
        fprintf(fid, '%s', char(hor_tit(b)));
        if b < length(hor_tit)
            fprintf(fid, '%s', ' & ');
        end
    end
    fprintf(fid, '%s', ' \\');
    fprintf(fid, '%s', ' \hline ');
    for b = 4:6
        fprintf(fid, '%s', char(left_name(b-3)));
        for c = 1:4
            fprintf(fid, '%.4f', tableSimulation(b,c));
            if c < 4
                fprintf(fid, '%s', ' & ');
            end
        end
        fprintf(fid, '%s', ' \\');
    end
    fclose(fid)

    fid = fopen('SimulationTable75.txt','w');
    for b = 1:length(hor_tit);
        fprintf(fid, '%s', char(hor_tit(b)));
        if b < length(hor_tit)
            fprintf(fid, '%s', ' & ');
        end
    end
    fprintf(fid, '%s', ' \\');
    fprintf(fid, '%s', ' \hline ');
    for b = 7:9
        fprintf(fid, '%s', char(left_name(b-6)));
        for c = 1:4
            fprintf(fid, '%.4f', tableSimulation(b,c));
            if c < 4
                fprintf(fid, '%s', ' & ');
            end
        end
        fprintf(fid, '%s', ' \\');
    end
    fclose(fid)

    %table_constrainedWeights

    for r=1:length(MaxDD_DD_LR_2_Cs(1,:))
        if MaxDD_DD_LR_2_Cs(r,:) ~= 0 
            if r == 1
                fid = fopen('constrained25.txt','w');
            elseif r == 2
                fid = fopen('constrained50.txt','w');
            end

            fprintf(fid, '%s', '');
            fprintf(fid, '%s', ' & ')
            fprintf(fid, '%s', 'DD LR');
            fprintf(fid, '%s', ' & ')
            fprintf(fid, '%s', 'DD LR Cs');
            fprintf(fid, '%s', ' \\ ')
            fprintf(fid, '%s', ' \hline ')

            fprintf(fid, '%s', 'Avg Return');
            fprintf(fid, '%s', ' & ')
            fprintf(fid, '%.4f', mean(mean_expected_excess_log_returns_sim2_DD_LR(:,r)));
            fprintf(fid, '%s', ' & ')
            fprintf(fid, '%.4f', mean(mean_expected_excess_returns_sim2_DD_LR_Cs(:,r)));
            fprintf(fid, '%s', ' \\ ')

            fprintf(fid, '%s', 'Avg Maximum DD');
            fprintf(fid, '%s', ' & ')
            fprintf(fid, '%.4f', mean(MaxDD_sim2_DD_LR(:,r)));
            fprintf(fid, '%s', ' & ')
            fprintf(fid, '%.4f', mean(MaxDD_DD_LR_2_Cs(Index_Cs(1):Index_Cs(end),r)));
            fprintf(fid, '%s', ' \\ ')

            fclose(fid)
        end
    end

    %Table Rebalanced

    fid = fopen('rebalanced.txt','w');

    fprintf(fid, '%s', '');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'DD LR Never Rebalanced');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'DD LR Rebalanced 3Y');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'DD LR Rebalanced 1Y');
    fprintf(fid, '%s', ' \\ ')
    fprintf(fid, '%s', ' \hline ')

    fprintf(fid, '%s', 'Avg Return');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(ex_25_mean_log_excess_returns_sim2'));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(ex_25_reb3_mean_log_excess_returns_sim2'));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(ex_25_reb1_mean_log_excess_returns_sim2'));
    fprintf(fid, '%s', ' \\ ')

    fprintf(fid, '%s', 'Avg Maximum DD');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(ex_25_maxDD_sim2));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(ex_25_reb3_maxDD_sim2));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(ex_25_reb1_maxDD_sim2));
    fprintf(fid, '%s', ' \\ ')

    fclose(fid)

    %Asset Classes 6

    fid = fopen('var6.txt','w');

    fprintf(fid, '%s', '');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Cash');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Equity');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Bonds');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Commodities');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Real Estate');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Hedge Funds');
    fprintf(fid, '%s', ' \\ ')
    fprintf(fid, '%s', ' \hline ')

    fprintf(fid, '%s', 'Mean');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,1)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,2)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,3)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,4)));
    fprintf(fid, '%s', ' & ');
    fprintf(fid, '%.4f', mean(data(:,5)));
    fprintf(fid, '%s', ' & ');
    fprintf(fid, '%.4f', mean(data(:,6)));
    fprintf(fid, '%s', ' \\ ');

    fprintf(fid, '%s', ' \hline ');

    fprintf(fid, '%s', 'Std');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,1)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,2)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,3)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,4)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,5)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,6)));
    fprintf(fid, '%s', ' \\ ')
    correl = corr(data(:,1:6));
    fprintf(fid, '%s', ' \hline ')
    for i=1:6
        if i==1
            fprintf(fid, '%s', 'Corr &')
        else
            fprintf(fid, '%s', ' &')
        end
        for j=1:6
            fprintf(fid, '%.4f', correl(i,j));
            if j<6
                fprintf(fid, '%s', ' & ')
            end
        end
        fprintf(fid, '%s', ' \\ ')
    end
    fclose(fid)

    %Asset Classes 10

    fid = fopen('var10a.txt','w');

    fprintf(fid, '%s', '');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Cash US');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Equity US');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Bonds US');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Commodities US');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Real Estate US');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Hedge Funds US');

    fprintf(fid, '%s', ' \\ ')
    fprintf(fid, '%s', ' \hline ')

    fprintf(fid, '%s', 'Mean');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,1)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,2)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,3)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,4)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,5)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,6)));
    fprintf(fid, '%s', ' \\ ')

    fprintf(fid, '%s', ' \hline ');

    fprintf(fid, '%s', 'Std');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,1)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,2)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,3)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,4)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,5)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,6)));

    fprintf(fid, '%s', ' \\ ')
    correl = corr(data_10(:,1:10));
    fprintf(fid, '%s', ' \hline ')
    for i=1:10
        if i==1
            fprintf(fid, '%s', 'Corr &')
        else
            fprintf(fid, '%s', ' &')
        end
        for j=1:6
            fprintf(fid, '%.4f', correl(i,j));
            if j<6
                fprintf(fid, '%s', ' & ')
            end
        end
        fprintf(fid, '%s', ' \\ ')
    end
    fclose(fid)


    fid = fopen('var10b.txt','w');

    fprintf(fid, '%s', '');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Corporate Bonds US');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Corporate Bonds Foreign');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Equity Foreign');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%s', 'Private Equity US');

    fprintf(fid, '%s', ' \\ ')
    fprintf(fid, '%s', ' \hline ')

    fprintf(fid, '%s', 'Mean');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,7)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,8)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,9)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', mean(data(:,10)));
    fprintf(fid, '%s', ' \\ ')

    fprintf(fid, '%s', ' \hline ');

    fprintf(fid, '%s', 'Std');
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,7)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,8)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,9)));
    fprintf(fid, '%s', ' & ')
    fprintf(fid, '%.4f', std(data(:,10)));

    fprintf(fid, '%s', ' \\ ')
    fprintf(fid, '%s', ' \hline ')
    for i=1:10
        if i==1
            fprintf(fid, '%s', 'Corr &')
        else
            fprintf(fid, '%s', ' &')
        end
        for j=7:10
            fprintf(fid, '%.4f', correl(i,j));
            if j<10
                fprintf(fid, '%s', ' & ')
            end
        end
        fprintf(fid, '%s', ' \\ ')
    end
    fclose(fid)
    disp('End of section tables');
end

disp('This is the end...');


