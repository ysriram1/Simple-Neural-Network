function [sigPrime] = sigmoidPrime(z)
    % Takes z and returns the derivative of the sigmoid function
    sigPrime = sigmoid(z).*(1.0-sigmoid(z));
end

