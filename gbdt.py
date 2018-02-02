# gradient boosting with sklearn

from sklearn.ensemble import GradientBoostingClassifier
from utils import get_split_data
from sklearn.externals import joblib
from xgboost import XGBClassifier
Xtr, ytr, Xts, yts = get_split_data()
print ('sklearn gbdt')
mdl = GradientBoostingClassifier(n_estimators=20000, max_depth=2, verbose=1, learning_rate=.03)
print(mdl)
mdl.fit(Xtr, ytr)
print (mdl.score(Xtr, ytr))
print (mdl.score(Xts, yts))
joblib.dump(mdl,'__pycache__/gbdt.mdl')
# 0.832 best score gbdt with 500,5,.1
# 0.839 with 5000,3,.1
# 0.842 with 5000,2,.1

'''print ('xgb')
mdl = XGBClassifier(n_estimators=300, max_depth=5, silent=False)
mdl.fit(Xtr, ytr)
print (mdl.score(Xtr, ytr))
print (mdl.score(Xts, yts))'''
