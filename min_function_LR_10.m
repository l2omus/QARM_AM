%------------------------------  Minimized Function for Liability Relative Approach, 10 Asset classes  ----------------------------%
% This Function is the function which is minimized when we use the fmincon framework in the maximum drawdown liability relative
% approach with 10 asset classes. It takes a vector of weights as input an return the maximum drawdown for our simulated data. This 
% function is dedicated to assets only approach and therefore it use the expected_excess_log_returns. 
%----------------------------------------------------------------------------------------------------------------------------------%


function [dd] = min_function_LR(w)    
    expected_excess_log_returns_9 = evalin('base','expected_excess_log_returns_10'); % Pull the matrix of expected excess log returns for 10 asset classes from our base workspace.
    MaxDD = maxdrawdown_logr(log((w*expected_excess_log_returns_9')'+1));	% Compute the maximum drawdown associated to the given weights and expected excess log returns
    dd = MaxDD;  															% The maxdrawdown_logr is defined and described in the file maxdrawdown_logr.m
end

