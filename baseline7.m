[train_size train_features] = size(words_train);
[test_size ~] = size(words_test);

tree = fitctree(words_train,genders_train);

y = predict(tree,words_test);