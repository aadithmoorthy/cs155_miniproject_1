# gradient boosting with sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from utils import *
from sklearn.externals import joblib
from xgboost import XGBClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
#scale = StandardScaler()
Xtr, ytr, Xts, yts = get_split_data()
#scale.fit(Xtr)
#Xtr = scale.transform(Xtr)
#Xts = scale.transform(Xts)
print ('sklearn gbdt')
mdl = GradientBoostingClassifier(n_estimators=5000, max_depth=2, verbose=1, learning_rate=.1)
print(mdl)
mdl.fit(Xtr, ytr)
print (mdl.score(Xtr, ytr))
print (mdl.score(Xts, yts))
'''joblib.dump(mdl,'__pycache__/gbdt.mdl')

# best split: 0.842/3 with 5000,2,.1 (no preprocessing)
platt_scaling = LogisticRegression()
preds = mdl.predict_proba(Xtr)
print preds.shape, ytr.shape
platt_scaling.fit(preds,ytr)
print 'plat', platt_scaling.score(preds,ytr)


result_col_2 = mdl.predict(get_test_data())
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('gbdt_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction')

result = platt_scaling.predict_proba(mdl.predict_proba(get_test_data()))[:,1]
np.savetxt('gbdt_result_probabilities.txt', result, fmt="%g", delimiter=',')'''
'''print ('xgb')
mdl = XGBClassifier(n_estimators=300, max_depth=5, silent=False)
mdl.fit(Xtr, ytr)
print (mdl.score(Xtr, ytr))
print (mdl.score(Xts, yts))'''
