# linear with no l1 regularization

from sklearn.linear_model import RidgeClassifier
from utils import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

Xtr, ytr = get_unsplit_data()
Xts = get_test_data()
Xtr = np.concatenate((Xtr, np.power(Xtr,2), np.log(Xtr+.00000001),1/(Xtr+.1)), axis=1)
Xts = np.concatenate((Xts, np.power(Xts,2), np.log(Xts+.00000001), 1/(Xts+.1)), axis=1)
print ('loaded data')
mdl = RidgeClassifier(alpha=4, normalize=True)
print(mdl)
mdl.fit(Xtr, ytr)

print (mdl.score(Xtr, ytr))
#print (mdl.score(Xts, yts))
result_col_2 = mdl.predict(Xts)
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('nonlinear_ridge_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction', comments="")

# (split) 0.854 with power of 2, log +.00000001, 1/(Xtr+.1) and alpha = 3 and normalized
