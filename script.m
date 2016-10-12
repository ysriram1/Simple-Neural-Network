clear;
clc;

% Sample runs

% Iris data
iris = csvread('.\sampleData\iris.csv');
irisX = iris(:,1:4)';
irisY = iris(:,5:7)';

NNCalc(irisX,irisY,[4,20,3],100,10,0.1);

% MNIST data
load('.\sampleData\mnistTrn.mat')
mnistX = trn;
mnistY = trnAns;

NNCalc(mnistX,mnistY,[784,30,10],30,10,3);

% XOR data
xor = csvread('.\sampleData\xor.csv');
xorX = xor(:,1:2)';
xorY = xor(:,3)';

NNCalc(xorX,xorY,[2,2,1],10,4,0.1);
NNCalc(xorX,xorY,[2,2,1],10,1,0.1);
NNCalc(xorX,xorY,[2,3,2,1],20,1,0.1);
NNCalc(xorX,xorY,[2,2,1],1000,5,0.1);