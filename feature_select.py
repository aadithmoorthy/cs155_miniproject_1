# feature select with lasso

from sklearn.linear_model import Lasso
from utils import get_split_data

Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')
mdl = Lasso(alpha=0.001)
mdl.fit(Xtr, ytr)

print (mdl.score(Xts, yts))
print (mdl.sparse_coef_.getnnz())
