filename2 = 'test_final_labelled.csv';
test_tbl = readtable(filename2);

[yfit,scores] = trainedModel.predictFcn(test_tbl);
%%
idx = 1:length(yfit);
output = [idx' double(yfit)];
output = [["ID" "Prediction"]; output];
writematrix(output,'results_opt_tree.csv')