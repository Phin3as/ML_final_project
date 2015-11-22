[train_size train_features] = size(words_train);
[test_size ~] = size(words_test);

%removing similar words
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

%normalizing data
for i=1:train_size
    m=rms(X_train(i,:));
    X_train(i,:)=X_train(i,:)./m;
end

for i=1:X_test
    m=rms(X_test(i,:));
    X_test(i,:)=X_test(i,:)./m;
end

%mdl = svmtrain(genders_train,X_train,'KernelFunction','kernel_intersection');
%tmp=zeros(test_size,1);
%[Y_test, accuracy, prob_estimates] = svmpredict(tmp,X_test,mdl);
func_i= @(X,X2) kernel_intersection(X, X2);

[test_err mdl]=kernel_libsvm(X_train,genders_train,X_test,tmp,func_i);

%svm_mdl = fitcsvm(words_train,genders_train,'KernelFunction','polynomial','PolynomialOrder',3);
%Y_test2 = predict(svm_mdl,words_test);