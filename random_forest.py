# random forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils import *
import numpy as np

Xtr, ytr = get_unsplit_data()
print ('loaded data')
mdl = RandomForestClassifier(n_estimators=1000, verbose=1, max_depth=50, n_jobs=4)
print mdl
mdl.fit(Xtr, ytr)

print (mdl.score(Xtr, ytr))
#print (mdl.score(Xts, yts))
# best split with n_estimators=1000, maxdepth=50, got .831
# best unsplit with n_estimators=1000, maxdepth=50, got .82780

platt_scaling = LogisticRegression()
preds = mdl.predict_proba(Xtr)
print preds.shape, ytr.shape
platt_scaling.fit(preds,ytr)
print 'plat', platt_scaling.score(preds,ytr)


result_col_2 = mdl.predict(get_test_data())
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('rf_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction', comments='')

result = platt_scaling.predict_proba(mdl.predict_proba(get_test_data()))[:,1]
np.savetxt('rf_result_probabilities.txt', result, fmt="%g", delimiter=',')
