%---------------------------------------------------  Generate Weights Function  --------------------------------------------------%
% This function doesn't require any input and return a vector rw of six elements between 0 and 1. The sum of the element will be 
% greater than 0.99 and smaller than 1.01
%----------------------------------------------------------------------------------------------------------------------------------%

function [rw] = generateWeights   
    w = [0 0 0 0 0 0]; % Create an empty array to store the weights
    n = [0.5:0.1:1.3]; % Vector of possible amount of weight for the first distribution
    a = n(randi(9,1,1)); % a is the base amount of weight to be distributed in the next iteration, we choose randomly a value in n for the first one.
    j = 0; % Initialize a counter for iterations
    while sum(w) < 0.99
        j = j+1;  % j counts the number of iterations
        i = randi(6,1,1); % Assign randomly an integer from 1 to 6 to the variable i
        w(i) = w(i) + rand*a; % Assign a random fraction of the amount a to the weight number i
        if j < 900  % While the number of iteration j is smaller than 900, we decrease a by 25% for the next iteration
            a = a/1.25;
        elseif j > 1000 % If the number of iteration j is larger than 1000, we increase a by 25% for the next iteration
            a =  a*1.25;
        end
        if sum(w) > 1.01 % If sum of the weights is larger than 1, we reset the variables and restart the process 
            a = n(randi(9,1,1));
            w = [0 0 0 0 0 0];
            j = 0;
        end
    end 
    rw = w; % Once the number is between 0.99 and 1.01, we return the vector of weigths
end

