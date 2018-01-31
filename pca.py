# PCA features getting

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from utils import get_split_data

Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')
mdl = PCA(n_components=500)
mdl.fit(Xtr)

Xtr_transform = (mdl.transform(Xtr))
Xts_transform = (mdl.transform(Xts))
print (Xts_transform.shape)

# followed by sgd test:
mdl = SGDClassifier(alpha=0.00001, max_iter=1000)
mdl.fit(Xtr_transform, ytr)

print (mdl.score(Xtr_transform, ytr))
print (mdl.score(Xts_transform, yts))
