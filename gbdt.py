# gradient boosting with sklearn

from sklearn.ensemble import GradientBoostingClassifier
from utils import get_split_data
from xgboost import XGBClassifier
Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')
mdl = GradientBoostingClassifier(n_estimators=500, max_depth=5, verbose=1)
mdl.fit(Xtr, ytr)

print (mdl.score(Xts, yts))

# 0.832 best score gbdt with 500,5


print ('loaded data')
mdl = XGBClassifier(n_estimators=500, max_depth=5, verbose=1)
mdl.fit(Xtr, ytr)

print (mdl.score(Xts, yts))
