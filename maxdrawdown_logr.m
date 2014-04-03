%---------------------------------------------------  Maxdrawdown_logr Function  --------------------------------------------------%
% This function takes a vector of log returns as input and return the maximum drawdown for the serie. It also return the periods
% associated to the maximum drawdown. The function maxdrawdown with option 'arithmetic' return the difference between the greater 
% and smaller element of a vector. To get the maxDD of our series if log returns, we need to transform them in cumulative log
% returns. Thereafter we can use the function maxdrawdown with option 'arithmetic' on our vector of cumulative log returns to get 
% the maximal drawdown of our serie of log returns.
%----------------------------------------------------------------------------------------------------------------------------------%

function [maxDD, periods] = maxdrawdown_logr(v)
   if length(v(:,1)) == 1 % We add a zero at the first position of the vector (log return for period zero is zero)
       v = [0 v];
   elseif length(v(1,:)) == 1
       v = [0; v];
   end
   [maxDD, periods] = maxdrawdown(cumsum(v), 'arithmetic'); % The function maxdrawdown with option 'arithmetic' on our vector of 
   															% cumulative log returns returns the maximal drawdown of our serie of log returns.
end

