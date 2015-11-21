[train_size train_features] = size(words_train);
[test_size ~] = size(words_test);

%X = [words_train;words_test];

%numpc=506;

svm_mdl = fitcsvm(words_train,genders_train,'KernelFunction','kernel_intersection','Prior','uniform');
Y_test = predict(svm_mdl,words_test);

%[evectors pv evalues] = pca(X);

%X_train = pv(1:train_size,1:numpc);
%X_test = pv(1+train_size:end,1:numpc);

%svm_pca_mdl = fitcsvm(X_train,genders_train,'KernelFunction','kernel_intersection');
%Y_test = predict(svm_pca_mdl,X_test);