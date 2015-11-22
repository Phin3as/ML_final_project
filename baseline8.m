[train_size train_features] = size(words_train);
[test_size ~] = size(words_test);

X=[words_train;words_test];

[a pv b] = pca(X);
numpc = 2500;

X_train = pv(1:train_size,1:numpc);
Y_train = genders_train(1:train_size,:);
X_test = pv(train_size+1:end,1:numpc);

X_train=words_train;
Y_train=genders_train;
X_test=words_test;

lm_mdl = fitlm(X_train,Y_train);

Y_test = predict(lm_mdl,X_test);

for i=1:numel(Y_test)
    if Y_test(i)<0.5
        Y_test(i)=0;
    else
        Y_test(i)=1;
    end
end