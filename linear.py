# linear with no l1 regularization

from sklearn.linear_model import RidgeClassifier
from utils import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
Xtr, ytr, Xts, yts = get_split_data()

Xtr_temp = np.zeros((19000,500*499/2+500))
Xts_temp = np.zeros((1000,500*499/2+500))
currIdx = 0
for i in tqdm(range(500)):
    for j in range(i, 500):
        Xtr_temp[:,currIdx] = np.minimum(Xtr[:,i], Xtr[:,j])
        Xts_temp[:,currIdx] = np.minimum(Xts[:,i], Xts[:,j])
        currIdx += 1

Xtr = Xtr_temp
Xts = Xts_temp

#Xtr = np.concatenate((Xtr, np.power(Xtr,2), np.log(Xtr+.00000001),1/(Xtr+.1)), axis=1)
#Xts = np.concatenate((Xts, np.power(Xts,2), np.log(Xts+.00000001), 1/(Xts+.1)), axis=1)
print ('loaded data')
mdl = RidgeClassifier(alpha=4, normalize=True)
print(mdl)
mdl.fit(Xtr, ytr)

print (mdl.score(Xtr, ytr))
print (mdl.score(Xts, yts))
'''result_col_2 = mdl.predict(Xts)
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('nonlinear_ridge_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction', comments="")
'''
# (split) 0.854 with power of 2, log +.00000001, 1/(Xtr+.1) and alpha = 3 and normalized
