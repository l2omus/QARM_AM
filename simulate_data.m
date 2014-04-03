%---------------------------------------------------  Simulate Data Function  -----------------------------------------------------%
% This function is used to perform a simulation and define expected log returns for our variables. It takes as input the matrix of
% historical data, the Psi_0 and Psi_0, and standard deviations of the error terms of a VAR(1) process performed on our data, the 
% number of years to simulate and the number of variables.
%----------------------------------------------------------------------------------------------------------------------------------%

function [sim_data] = simulate_data(data, Psi_0, Psi_1, std_dev_res, T, N) 

sim1_data(1,1:N) =  Psi_0 + data(end,:)*Psi_1; % Use the parameters of the VAR to build expected data

T = T*4;
N = length(Psi_0);

for t=1:T-1
    for i=1:N;
        error_terms(t,i) = normrnd(0, std_dev_res(i)); % Create an innovation term for each variable
    end
    sim1_data(t+1,1:N) =  (Psi_0 + sim1_data(t,:)*Psi_1 + error_terms(t,:)); % Add the innovation term
end

corr_sim1 = corr(sim1_data(:,1:end-3)); % Compute the correlation matrix of the simulated data

for t=1:length(sim1_data)-8 % Add probability of 1% that a crash during 8 quarters starts
    if rand <= 0.01
        for c=0:7
            for j=1:7
                sim1_data(t+c,j) = sim1_data(t+c,j)- 0.5*std_dev_res(1,2)*corr_sim1(2,j); % The crash impact negatively the expected log excess returns proportionally 
            end                                                                           % to the correlation of each asset to the asset class equity
        end
        disp('A crash has been generated!');
    end
end

for i=1:T/4
    sim_data_red(i,:) = sum(sim1_data(4*i-3:4*i,:)); % Compute the expected values in term of years
end

sim_data = sim_data_red; % Return the expected values
end
