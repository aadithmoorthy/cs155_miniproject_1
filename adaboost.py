# gradient boosting with sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from utils import get_split_data


Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')
mdl = AdaBoostClassifier(n_estimators=500, learning_rate=0.3, base_estimator=DecisionTreeClassifier(max_depth=1))
mdl.fit(Xtr, ytr)

print (mdl.score(Xts, yts))
# with n_estimators=500 -> got 0.832, w/ dt, lr=0.3
