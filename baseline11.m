[train_size train_features] = size(words_train);
[test_size ~] = size(words_test);

%males = (genders_train==0);
%females = (genders_train==1);
%males_matrix = words_train(males,:);
%females_matrix = words_train(females,:);

%males_vector=mean(males_matrix);
%females_vector=mean(females_matrix);

%diff = males_vector-females_vector;

%features = (diff~=0);

%X_train = words_train(:,features);
%X_test = words_test(:,features);

addpath('E:/Masters/ML/HW7/hw7_kit/hw7_kit/liblinear');

test_y=zeros(test_size,1);
Y_test=logistic(words_train, genders_train, words_test, test_y);