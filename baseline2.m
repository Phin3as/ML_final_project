[train_size ~] = size(words_train);
[test_size ~] = size(words_test);

X_full = [words_train;words_test];


[ev pv eigenvalues] = pca(X_full);

ev_size=numel(eigenvalues);
numpc=1000;
X_train = pv(1:train_size,1:numpc);
X_test = pv(train_size+1:end,1:numpc);

cos_sim = @(x,y) dot(x,y)./(norm(x,2)*norm(y,2));


op=zeros(test_size,1);
for i=1:test_size
    i
    new_person = X_test(i,:);
    cos_sim_matrix=zeros(train_size,1);
    for j=1:train_size
        old_person=X_train(j,:);
        cos_sim_matrix(j)=cos_sim(new_person,old_person);
    end
    [sajal,sajal_idx] = sort(cos_sim_matrix,'descend');
    idx = sajal_idx(1:20,:);
    gender = mean(genders_train(idx,:));
    if gender > 0.5
        gender=1;
    else
        gender=0;
    end
    op(i)=gender;
end