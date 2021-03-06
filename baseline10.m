[train_size train_features] = size(words_train);
[test_size ~] = size(words_test);

males = (genders_train==0);
females = (genders_train==1);
males_matrix = words_train(males,:);
females_matrix = words_train(females,:);

males_vector=mean(males_matrix);
females_vector=mean(females_matrix);

diff = males_vector-females_vector;

features = (diff~=0);

X_train = words_train(:,features);
X_test = words_test(:,features);

svm_mdl = fitcsvm(X_train,genders_train,'KernelFunction','kernel_intersection');
Y_test = predict(svm_mdl,X_test);