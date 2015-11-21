[train_size train_features] = size(words_train);
[test_size ~] = size(words_test);

mdl = svmtrain(genders_train,words_train);

tmp=zeros(test_size,1);

[predicted_label, accuracy, prob_estimates] = svmpredict(tmp,words_test,mdl);
