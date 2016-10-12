function [sig] = sigmoid(z)
    % Takes z and returns the sigmoid of it. z could be a vector.
    sig = 1.0./(1.0+exp(-z));
end

