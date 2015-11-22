%Cosine Similarity Acc 76%
cos_sim = @(x,y) dot(x,y)./(norm(x,2)*norm(y,2));

[train_size ~] = size(words_train);
[test_size ~] = size(words_test);

op=zeros(test_size,1);
for i=1:test_size
    new_person = words_test(i,:);
    cos_sim_matrix=zeros(train_size,1);
    for j=1:train_size
        old_person=words_train(j,:);
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
