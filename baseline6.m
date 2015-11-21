[train_size train_features] = size(words_train);
[test_size ~] = size(words_test);

ind = crossvalind('Kfold',train_size , 10);

X_train = pv((ind~=2),:);
Y_train = genders_train((ind~=2),:);
X_test = pv((ind==2),:);
Y_test = genders_train((ind==2),:);

lambda=0.00001;

w_hat = ridge(Y_train,X_train,lambda);
y_hat = X_test*w_hat;
for i=1:numel(y_hat)
    if y_hat(i)<0.5
        y_hat(i)=0;
    else
        y_hat(i)=1;
    end
end

diff = Y_test-y_hat;
acc = sum(diff==0)./numel(diff)
