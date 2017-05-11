clear
clc
close all

% Loading the example dataset

%load fisheriris
%X = meas;

%Y = [ones(100,1); -1 * ones(50,1)];

%Y = [ones(50,1); -1 * ones(50,1);ones(50,1)];

M = csvread('H:/Datamining/UCI/Segment/segmentation.csv');
X = M( : ,1:end -1);
Y = M( : ,end);

[N D] = size(X);
% Standardize the feature sapce
for i = 1:D
    X_scaled(:,i) = 2*((X(:,i) - min(X(:,i))) / ( max(X(:,i)) - min(X(:,i)) ))-1;
end
X_scaled = X_scaled + normrnd(0,0.01,size(X_scaled));

NumberFolds = 3;
NumIteration = 2;

SR_RG = 1;
stepSize = 1;

division = round(N/NumberFolds);

%% Buiding the models
for ite = 1:NumIteration
    C = cvpartition(Y,'k',NumberFolds);
    for num = 1:NumberFolds;
        trainData = X(training(C,num),:);
        trainLabel = Y(training(C,num),:);
        testData = X(test(C,num),:);
        testLabel = Y(test(C,num),:);
        %% test data
        testData = cat(2,testData,testLabel);
        %csvwrite('test.csv', testData);
        csvwrite('testData.csv', testData);
        csvwrite('testLabel.csv', testLabel);
        %% Oversampling using SMOTE
        display ('SMOTE:')
        [trainDataSMOTE, trainLabelSMOTE] = SMOTE(trainData,trainLabel);
        %trainSMOTE = cat(2,trainDataSMOTE,trainLabelSMOTE);
        csvwrite('trainDataSMOTE.csv', trainDataSMOTE);
        csvwrite('trainLabelSMOTE.csv', trainLabelSMOTE);
        %% Oversampling using Borderline SMOTE
        display ('Borderline SMOTE:')
        NNC = 5;
        [borderMin_BorSMOTE, trainDatanewBorSMOTE, trainLabelnewBorSMOTE] = BorSMOTE(trainData,trainLabel,NNC);
        trainBorSMOTE = cat(2,trainDatanewBorSMOTE,trainLabelnewBorSMOTE);
        %csvwrite('BorSMOTE.csv', trainBorSMOTE);
        csvwrite('trainDatanewBorSMOTE.csv', trainDatanewBorSMOTE);
        csvwrite('trainLabelnewBorSMOTE.csv', trainLabelnewBorSMOTE);
        %% Oversampling using Safe-level SMOTE
        display ('Safe-level SMOTE:')
        NNC = 5;
        [trainDatanewSafeSMOTE, trainLabelnewSafeSMOTE] = Safe_Level_SMOTE(trainData,trainLabel,NNC);
        train_newSafeSMOTE = cat(2,trainDatanewSafeSMOTE,trainLabelnewSafeSMOTE);
        %csvwrite('train_newSafeSMOTE.csv', train_newSafeSMOTE);
        csvwrite('trainDatanewSafeSMOTE.csv', trainDatanewSafeSMOTE);
        csvwrite('trainLabelnewSafeSMOTE.csv', trainLabelnewSafeSMOTE);
        %% Oversampling using ASUWO
        display ('ASUWO:')
        CThresh = 3;
        K = 3;
        NN = 8;
        NS = 8;
        [trainDatanewASUWO, trainLabelnewASUWO] = ASUWO(trainData,trainLabel, CThresh , K, NN, NS);
        trainnewASUWO = cat(2,trainDatanewASUWO,trainLabelnewASUWO);
        %csvwrite('trainnewASUWO.csv', trainnewASUWO);
        csvwrite('trainnewASUWO.csv', trainDatanewASUWO);
        csvwrite('trainLabelnewASUWO.csv', trainLabelnewASUWO);
    end
    perm = [];
end