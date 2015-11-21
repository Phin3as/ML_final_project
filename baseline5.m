[train_size train_features] = size(words_train);
[test_size ~] = size(words_test);

ind = crossvalind('Kfold',train_size , 10);

[a pv b] = pca(words_train);
numpc = 3000;

X_train = pv((ind~=2),1:numpc);
Y_train = genders_train((ind~=2),:);
X_test = pv((ind==2),1:numpc);
Y_test = genders_train((ind==2),:);

%svm_mdl = fitcsvm(words_train,genders_train,'KernelFunction','kernel_intersection');

%svm_pca_mdl = fitcsvm(words_train,genders_train,'KernelFunction','polynomial','PolynomialOrder',2);
%Y_test2 = predict(svm_pca_mdl,words_test);

svm_pca_mdl = fitcsvm(X_train,Y_train,'KernelFunction','kernel_intersection');
Y_test1 = predict(svm_pca_mdl,X_test);


%nb_pca_mdl = fitcnb(words_train,genders_train);
%Y_test2 = predict(nb_pca_mdl,words_test);

%knn_pca_mdl = fitcknn(X_train,Y_train,'NumNeighbors',20);
%Y_test3 = predict(knn_pca_mdl,X_test);

diff = Y_test-Y_test1;

acc = sum(diff==0)./numel(diff)

%svm_pca_mdl = fitcsvm(words_train,genders_train,'KernelFunction','rbf');
%Y_test3 = predict(svm_pca_mdl,words_test);

%Y_test=Y_test1+Y_test2+Y_test3;
%Y_test=Y_test./3;

%op=zeros(numel(Y_test),1);

%for i=1:test_size
%    gender=Y_test(i);
%    if gender > 0.5
%        gender=1;
%    else
%        gender=0;
%    end
%    op(i)=gender;
%end