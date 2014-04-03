%--------------------------------------------------  Function ret2tick for log Returns  -------------------------------------------%
% This function takes a vector of log return as input and return the corresponding prices serie starting at price 1.
%----------------------------------------------------------------------------------------------------------------------------------%

function [prices] = ret2tick_log_returns(v)
    v = cumsum(v);
    p = [zeros(size(v(1,:))); v];
    p = exp(p);
    prices = p;
end