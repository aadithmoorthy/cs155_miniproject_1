# naive linear with SGD

from sklearn.linear_model import SGDClassifier
from utils import get_split_data

Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')
mdl = SGDClassifier(alpha=0.00001, max_iter=1000, verbose=1, penalty='elasticnet', l1_ratio=0.3)
mdl.fit(Xtr, ytr)

print (mdl.score(Xtr, ytr))
print (mdl.score(Xts, yts))
# 0.845 best score with a=0.00001, max_iter=1000, l1_ratio = 0.3
