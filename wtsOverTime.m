%------------------------------------------------ Function weights over time ------------------------------------------------------%
% This function calculate the evolution of the weights of a portfolio according to the expected excess log return for different 
% strategies of rebalancing. It returns the evolution of the weights for a portfolio rebalanced yearly, every three years and never
% rebalanced. It takes as input the intial weights of the optimal portfolio and the expected excess log returns.
%----------------------------------------------------------------------------------------------------------------------------------%

function [w w3 w1] = wtsOverTime(wts_optPort, expected_excess_log_returns)

    T = length(expected_excess_log_returns(:,1));
    N_assets = length(wts_optPort);
    wts_DD_LR_overtime = zeros(T , N_assets); % Create arrays of zeros to store future data...
    wts_DD_LR_overtime_reb1 = zeros(T , N_assets);
    wts_DD_LR_overtime_reb3 = zeros(T , N_assets);

    for t=1:T
        for j=1:N_assets
            if t==1     % Compute the evolution of the weights if we never rebalance the portfolio
                wts_DD_LR_overtime(t,j) = (wts_optPort(1,j)*(1+expected_excess_log_returns(t,j)))/...       
                    (wts_optPort(1,:)*(1+expected_excess_log_returns(t,:)'));
            else
                wts_DD_LR_overtime(t,j) = (wts_DD_LR_overtime(t-1,j)*(1+expected_excess_log_returns(t,j)))/...
                    (wts_DD_LR_overtime(t-1,:)*(1+expected_excess_log_returns(t,:)'));
            end
            N_reb = 3:3:30;
            if t==1     % Compute the evolution of the weights if we rebalance the portfolio every 3 years
                wts_DD_LR_overtime_reb3(t,j) = (wts_optPort(1,j)*(1+expected_excess_log_returns(t,j)))/...
                        (wts_optPort(1,:)*(1+expected_excess_log_returns(t,:)'));
            elseif ismember(t, N_reb)
                wts_DD_LR_overtime_reb3(t,j) = wts_optPort(1,j);
            else
                wts_DD_LR_overtime_reb3(t,j) = (wts_DD_LR_overtime_reb3(t-1,j)*(1+expected_excess_log_returns(t,j)))/...
                        (wts_DD_LR_overtime_reb3(t-1,:)*(1+expected_excess_log_returns(t,:)'));
            end
            clear N_reb;
            N_reb = 1:1:30;
            if t==1 % Compute the evolution of the weights if we rebalance the portfolio every year
                wts_DD_LR_overtime_reb1(t,j) = (wts_optPort(1,j)*(1+expected_excess_log_returns(t,j)))/...
                        (wts_optPort(1,:)*(1+expected_excess_log_returns(t,:)'));
            elseif ismember(t, N_reb)
                wts_DD_LR_overtime_reb1(t,j) = wts_optPort(1,j);
            else
                wts_DD_LR_overtime_reb1(t,j) = (wts_DD_LR_overtime_reb1(t-1,j)*(1+expected_excess_log_returns(t,j)))/...
                        (wts_DD_LR_overtime_reb1(t-1,:)*(1+expected_excess_log_returns(t,:)'));
            end
        end
    end
    w = wts_DD_LR_overtime;
    w3 = wts_DD_LR_overtime_reb3;
    w1 = wts_DD_LR_overtime_reb1;
end