# gradient boosting with sklearn

from sklearn.ensemble import GradientBoostingClassifier
from utils import get_split_data

Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')
mdl = GradientBoostingClassifier(n_estimators=500, max_depth=5, verbose=1)
mdl.fit(Xtr, ytr)

print (mdl.score(Xts, yts))

# 0.832 best score with 500,5
