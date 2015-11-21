[train_size ~] = size(words_train);
[test_size ~] = size(words_test);

X_full = [words_train;words_test];

k=20;

[ev pv eigenvalues] = pca(X_full);

ev_size=numel(eigenvalues);
numpc=2500;

X_train = pv(1:train_size,1:numpc);
X_test = pv(train_size+1:end,1:numpc);

cos_sim = @(x,y) dot(x,y)./(norm(x,2)*norm(y,2));

cluster_idx = kmeans(X_train,k);
clusters=zeros(k,numpc);
labels=zeros(k,1);

for i=1:k
    cluster_data = (cluster_idx==i);
    actual_cluster_data = X_train(cluster_data,:);
    clusters(i,:) = mean(actual_cluster_data(:,:));
    labels(i,:) = mean(genders_train(cluster_data,:));
end

for i=1:test_size
    new_person = X_test(i,:);
    cos_sim_matrix=zeros(numel(labels),1);
    for j=1:k
        old_cluster = clusters(j,:);
        cos_sim_matrix(j)=cos_sim(old_cluster,new_person);
    end
    [sajal,sajal_idx] = sort(cos_sim_matrix,'descend');
    idx = sajal_idx(1:3,:); %num of closest clusters to pick
    gender = mean(labels(idx,:));
    if gender > 0.5
        gender=1;
    else
        gender=0;
    end
    op(i)=gender;
end