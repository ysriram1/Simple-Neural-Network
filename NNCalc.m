function NNCalc(inputs, targets, nodeLayers, numEpochs, batchSize, eta)
    % inputs: matrix with rows as features and columns as instances
    % targets: matrxi with rows as labels and columns as instances
    % nodeLayers: array with number of neurons per layer including input, output
    %       example: [2,3,4] means 2 input neurons, 1 hidden layer with 3
    %       neurons and 4 output neurons
    % numEpochs: number of epochs to run for
    % batchSize: size of mini-batch
    % eta: learning rate
    % loop through the layers to initialize the weights
    layerCount = length(nodeLayers);
    for layerIndex=2:layerCount;
        W{layerIndex} = normrnd(0,1, [nodeLayers(layerIndex),nodeLayers(layerIndex - 1)]); 
        b{layerIndex} = normrnd(0,1, [nodeLayers(layerIndex),1]);
    end

    % n is number of total number of training cases and f is number of features
    % m is mini batch size
    [f n] = size(inputs); 
    [classCount cases] = size(targets);
    for iEpoch=1:numEpochs
        m = batchSize;
        count = 1;
        while count < n 
            %if last batch exceeds size of dataset, then reduce the batch size
            if count + m > n
                m = n - count + 1;
            end       
            % Note that we feed in the entire batch at once.
            % Forward propagation
            a{1}=inputs(:,count:count+m-1); % this is the input batch
            for layer=2:layerCount
                [aRows aCols] = size(a{layer-1});
                z{layer} = W{layer}*a{layer-1}+repmat(b{layer},1,aCols);
                a{layer} = sigmoid(z{layer});
            end
            
            % Error calculation
            costDerivative = a{layerCount}-targets(:,count:count+m-1);
            error{layerCount} = costDerivative .* sigmoidPrime(z{layerCount}); %output error 
            
            % Back propagation
            for layer=layerCount-1:-1:2
                error{layer} = (W{layer+1}'*error{layer+1}).*sigmoidPrime(z{layer});
            end
            
            % SGD
            for layer=layerCount:-1:2
                W{layer} = W{layer} - (eta/m)*(error{layer}*a{layer-1}'); 
                b{layer} = b{layer} - (eta/m)*sum(error{layer},2);
            end
            count = count + m; 
        end
        
        % Calculating accuracy and MSE on the entire data
        a{1} = inputs; Y = targets;
        for layer=2:layerCount
            z{layer} = W{layer}*a{layer-1}+repmat(b{layer},1,n);
            a{layer} = sigmoid(z{layer});
        end
        
        % the accuracy
        % set the largest output val to 1 and the rest to 0
        Y_pred = bsxfun(@eq, a{layerCount}, max(a{layerCount}, [], 1));
        if classCount == 1 % if the output is 1 in length and binary
            Y_pred = round(a{layerCount});
        end
        checkMat = sum(Y_pred == Y,1) == classCount;
        corClass = sum(checkMat(:));
        accuracy = corClass/n;
        %the MSE
        mse = immse(a{layerCount},Y);
        
        fprintf('Epoch %i, MSE:%f, Correct: %i/%i, Acc: %f\n', iEpoch, mse, corClass, n, accuracy);
        if accuracy == 1
            break;
        end
    end
end