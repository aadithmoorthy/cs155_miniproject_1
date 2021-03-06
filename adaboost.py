# gradient boosting with sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from utils import *
import numpy as np

Xtr, ytr = get_unsplit_data()
print ('loaded data')
mdl = AdaBoostClassifier(n_estimators=500, learning_rate=0.3, base_estimator=DecisionTreeClassifier(max_depth=2))
print mdl
mdl.fit(Xtr, ytr)

print (mdl.score(Xtr, ytr))
#print (mdl.score(Xts, yts))
# with n_estimators=500 -> got 0.833, w/ dt, lr=0.3 maxdepth=2 (fully studied all maxdepth=2 and this is best)

platt_scaling = LogisticRegression()
preds = mdl.predict_proba(Xtr)
print preds.shape, ytr.shape
platt_scaling.fit(preds,ytr)
print 'plat', platt_scaling.score(preds,ytr)


result_col_2 = mdl.predict(get_test_data())
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('adaboost_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction', comments='')

result = platt_scaling.predict_proba(mdl.predict_proba(get_test_data()))[:,1]
np.savetxt('adaboost_result_probabilities.txt', result, fmt="%g", delimiter=',')
