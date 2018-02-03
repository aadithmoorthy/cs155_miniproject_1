# knn.py

# logistic linear

from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from utils import *
import numpy as np
from sklearn.metrics import accuracy_score
Xtr, ytr, Xts, yts = get_split_data()

print ('loaded data')
mdl = RadiusNeighborsClassifier(radius=50, outlier_label=2)

mdl.fit(Xtr, ytr)


#print (mdl.score(Xtr, ytr))
try:
    pred = mdl.predict(Xts)
except ValueError:
    mdl2 = KNeighborsClassifier(n_neighbors=20)
    mdl2.fit(Xtr, ytr)
    pred2 = mdl2.predict(Xts)
    pred[np.where(pred==2)] = pred2[np.where(pred==2)]

print (accuracy_score(pred,yts))
#print (mdl.predict(Xts))
#print (mdl.predict_proba(Xts))

# bests:

'''result_col_2 = mdl.predict(get_test_data())
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('knn_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction')

merge_result, result = probabilities(Xts,yts,get_test_data())
np.savetxt('knn_result_probabilities.txt', result, fmt="%g", delimiter=',')

np.savetxt('knn_merge_result.txt', merge_result, fmt="%g", delimiter=',')
print (mdl.score(Xts, yts))'''
