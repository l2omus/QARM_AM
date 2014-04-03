%---------------------------------------------------  Minimized Function for Assets Only --------------------------------------------%
% This Function is the function which is minimized when we use the fmincon framework in the maximum drawdown assets only approach.
% It takes a vector of weights as input an return the maximum drawdown for our simulated data. This function is dedicated to assets
% only approach and therefore it use the expected_log_returns. 
%----------------------------------------------------------------------------------------------------------------------------------%

function [dd] = min_function_AO(w) % 
    expected_log_returns = evalin('base','expected_log_returns'); 	% Pull the matrix of expected log returns from our base workspace
    MaxDD = maxdrawdown_logr(log((w*expected_log_returns')'+1)); 	% Compute the maximum drawdown associated to the given weights and expected log returns
    dd = MaxDD;  													% The maxdrawdown_logr is defined and described in the file maxdrawdown_logr.m
end