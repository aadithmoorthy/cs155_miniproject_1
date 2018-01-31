# Support vector machine
from sklearn.svm import SVC
from utils import get_split_data


Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')
mdl = SVC(verbose=True)
mdl.fit(Xtr, ytr)

print (mdl.score(Xts, yts))
