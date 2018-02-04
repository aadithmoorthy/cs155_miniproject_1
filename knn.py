# knn.py

# logistic linear

from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from utils import *
import numpy as np
from sklearn.metrics import accuracy_score
Xtr, ytr = get_unsplit_data()

print ('loaded data')
mdl = KNeighborsClassifier(n_neighbors=80)
print(mdl)
mdl.fit(Xtr, ytr)
print ('done fit')

#print (mdl.score(Xts, yts))

#print (accuracy_score(pred,yts))
#print (mdl.predict(Xts))
#print (mdl.predict_proba(Xts))

# bests: split 0.693 with 80

result_col_2 = mdl.predict(get_test_data())
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('knn_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction', comments="")

'''
np.savetxt('knn_merge_result.txt', merge_result, fmt="%g", delimiter=',')
print (mdl.score(Xts, yts))'''
