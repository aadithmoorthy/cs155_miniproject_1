# kernel ridge regression

from sklearn.kernel_ridge import KernelRidge
from utils import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
scale = StandardScaler()
Xtr, ytr = get_split_data()
Xts = get_test_data()
scale.fit(Xtr)
Xtr = scale.transform(Xtr)
Xts = scale.transform(Xts)
print ('loaded data')
start = datetime.now()
mdl = KernelRidge(alpha=0.03, kernel="rbf")
print(mdl)
mdl.fit(Xtr, ytr)
print('time elapsed to fit:', datetime.now()-start)
# polynomial best on split: alpha=60 & polynomial kernel deg 3 - 0.849
# rbf best on split: alpha=0.03 - 0.822
# laplacian best on split: alpha=3 - 0.848
print (accuracy_score(np.maximum(np.minimum(np.round(mdl.predict(Xtr)),1),0), ytr))
#print (accuracy_score(np.maximum(np.minimum(np.round(mdl.predict(Xts)),1),0), yts))
'''result_col_2 = np.maximum(np.minimum(np.round(mdl.predict(Xts)),1),0)
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('kernel_ridge_rbf_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction', comments="")
'''
