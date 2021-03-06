# naive bayes

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from scipy.sparse import coo_matrix
from utils import *
import numpy as np
from tqdm import tqdm
Xtr, ytr, Xts, yts = get_split_data()
'''Xtr_temp = np.zeros((19000,500*499/2+500))
Xts_temp = np.zeros((1000,500*499/2+500))
currIdx = 0
for i in tqdm(range(500)):
    for j in range(i, 500):
        Xtr_temp[:,currIdx] = np.minimum(Xtr[:,i], Xtr[:,j])
        Xts_temp[:,currIdx] = np.minimum(Xts[:,i], Xts[:,j])
        currIdx += 1

Xtr = Xtr_temp
Xts = Xts_temp'''
#scale.fit(Xtr)
#Xtr = scale.transform(Xtr)
#Xts = scale.transform(Xts)
print ('loaded data')
mdl = MultinomialNB(alpha=0)
mdl.fit(Xtr, ytr)

print (mdl.score(Xtr, ytr))
print (mdl.score(Xts, yts))
'''platt_scaling = LogisticRegression()
preds = mdl.predict_proba(Xtr)
print preds.shape, ytr.shape
platt_scaling.fit(preds,ytr)
print 'plat', platt_scaling.score(preds,ytr)
#print (mdl.predict(Xts))
#print (mdl.predict_proba(Xts))

# best on split - 0.848 with alpha = 0
# best on unsplit: ? accidentally ran on split - should do again

result_col_2 = mdl.predict_proba(get_test_data())
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('naive_bayes_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction')

result = platt_scaling.predict_proba(mdl.predict_proba(get_test_data()))[:,1]
np.savetxt('naive_bayes_result_probabilities.txt', result, fmt="%g", delimiter=',')
'''
