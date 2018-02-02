# model merging on val:
from utils import *
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#_, _, _, yval = get_split_data()
#val_pred1 = np.loadtxt('naive_linear_merge_result.txt').reshape((len(yval),1))
#val_pred2 = np.loadtxt('nn_merge_result.txt').reshape((len(yval),1))
#val_pred3 = np.loadtxt('logistic_linear_merge_result.txt').reshape((len(yval),1))
#regressors = np.concatenate((val_pred2, val_pred3), axis=1)

'''mdl = AdaBoostClassifier(n_estimators=10, base_estimator=DecisionTreeClassifier(max_depth=1))#LogisticRegression(C = 0.1)
# adaboost get to .85220 with 10 estimators and decision stumps
mdl.fit(regressors, yval)

print (accuracy_score(yval, np.round(mdl.predict(regressors))))
'''
test_num = 10000
'''

val_pred2 = np.loadtxt('nn_result_probabilities.txt').reshape((test_num,1))
val_pred3 = np.loadtxt('logistic_linear_result_probabilities.txt').reshape((test_num,1))
regressors = np.concatenate((val_pred2, val_pred3), axis=1)
result_col_2 = mdl.predict(regressors)
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('merge_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction')
'''
# also try averaging with leaderboard scores!
scores = [0.85120,0.85180]
sum_scores = np.sum(scores)
merge = np.zeros((test_num,1))
merge += (0.85120/sum_scores) * np.loadtxt('logistic_linear_result_probabilities.txt').reshape((test_num,1))
merge += (0.85180/sum_scores) * np.loadtxt('nn_result_probabilities.txt').reshape((test_num,1))
merge += (-/sum_scores) * np.loadtxt('naive_bayes_result_probabilities.txt').reshape((test_num,1))
result_col_2 = merge
print(merge, sum_scores)
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('merge_result.txt', np.round(result), fmt="%d", delimiter=',', header='Id,Prediction')
