# kernel ridge regression

from sklearn.kernel_ridge import KernelRidge
from utils import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
scale = StandardScaler()
Xtr, ytr, Xts, yts = get_split_data()

scale.fit(Xtr)
Xtr = scale.transform(Xtr)
Xts = scale.transform(Xts)
print ('loaded data')
mdl = KernelRidge(alpha=0.00001, kernel="polynomial")
print(mdl)
mdl.fit(Xtr, ytr)

print (accuracy_score(np.maximum(np.minimum(np.round(mdl.predict(Xtr)),1),0), ytr))
print (accuracy_score(np.maximum(np.minimum(np.round(mdl.predict(Xts)),1),0), yts))
'''result_col_2 = mdl.predict(Xts)
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('kernel_ridge_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction', comments="")
'''
