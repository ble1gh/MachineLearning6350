%ML final project submission #2
%Brook Leigh
clear all

%read in data and assign headers
filename = 'train_final.csv';
opts = detectImportOptions(filename);
train_tbl = readtable(filename,TextType="String");
predictorNames = {'age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label'};
train_tbl.Properties.VariableNames = predictorNames;

filename2 = 'test_final.csv';
opts2 = detectImportOptions(filename2);
test_tbl = readtable(filename2,TextType="String");
test_tbl = test_tbl(:,2:end);
predictorNames_test = {'age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country'};
test_tbl.Properties.VariableNames = predictorNames_test;

%change unknown values to the most common value in that category
for i = 1:numel(predictorNames_test)
    name = predictorNames_test(i);
    % Find the unique elements and their counts
    [uniqueElements, ~, indices] = unique(train_tbl(:,i));
    counts = accumarray(indices, 1);
    % Find the element with the maximum count
    [maxCount, maxIndex] = max(counts);
    mostFrequentElement = train_tbl(maxIndex,i);
    for j = 1:size(train_tbl,1)
        if string(train_tbl{j,i}) == "?"
            train_tbl(j,i) = mostFrequentElement;
        end
    end
    for j = 1:size(test_tbl,1)
        if string(test_tbl{j,i}) == "?"
            test_tbl(j,i) = mostFrequentElement;
        end
    end
end

%change output label to categorical
labelName = "label";
train_tbl = convertvars(train_tbl,labelName,"categorical");


%change qualitative variables to categorical 
categoricalPredictorNames = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"];
train_tbl = convertvars(train_tbl,categoricalPredictorNames,"categorical");
test_tbl = convertvars(test_tbl,categoricalPredictorNames,"categorical");

%change categorical variables to one-hot encoded
for i = 1:numel(categoricalPredictorNames)
    name = categoricalPredictorNames(i);
    train_tbl.(name) = onehotencode(train_tbl.(name),2);
    test_tbl.(name) = onehotencode(test_tbl.(name),2);
end

%split data into training, validation, and test partitions
numObservations = size(train_tbl,1);
[idxTrain,idxValidation] = trainingPartitions(numObservations,[0.80 0.2]);

tblTrain = train_tbl(idxTrain,:);
tblValidation = train_tbl(idxValidation,:);

%change predictor names to an array of strings
p_len = length(predictorNames);
predictorNames_array = strings(p_len,1);
for i = 1:p_len
    predictorNames_array(i) = string(predictorNames(i));
end

%create arrays of doubles for NN training
d = size(train_tbl);
XTrain = double.empty;
XValidation = double.empty;
XTest = double.empty;
for i = 1:(d(2)-1)
    XTrain = [XTrain double(table2array(tblTrain(:,i)))];
    XValidation = [XValidation double(table2array(tblValidation(:,i)))];
    XTest = [XTest double(table2array(test_tbl(:,i)))];
end

%because the training data has a missing country
XTrain = [XTrain zeros(size(XTrain,1),1)];
XValidation = [XValidation zeros(size(XValidation,1),1)];

%Output data
TTrain = onehotencode(tblTrain.label,2);
TValidation = onehotencode(tblValidation.label,2);

%%
%Define NN architecture
numFeatures = size(XTrain,2);
hiddenSize = 3;
numClasses = 2;
 
layers = [
    featureInputLayer(numFeatures,Normalization="zscore")
    fullyConnectedLayer(hiddenSize)
    layerNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer];

options = trainingOptions("lbfgs", ...
    ExecutionEnvironment="cpu", ...
    ValidationData={XValidation,TValidation}, ...
    ValidationFrequency=5, ...
    OutputNetwork="best-validation", ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false)
    InputDataFormats = "B";

%Train NN
net = trainnet(XTrain,TTrain,layers,"binary-crossentropy",options);

%%
%predict using NN
classNames = [0 1];
scoresTest = minibatchpredict(net,XTest);
YTest = onehotdecode(scoresTest,classNames,2);

%%
%add indexes to output and write to file
Y = double(YTest);
idx = 1:length(Y);
output = [idx' (Y-1)];
output = [["ID" "Prediction"]; output];
writematrix(output,'results_2.csv')
