# naive linear with SGD

from sklearn.linear_model import SGDClassifier
from utils import *
import numpy as np

Xtr, ytr = get_unsplit_data()
print ('loaded data')
mdl = SGDClassifier(alpha=0.00001, max_iter=1000, verbose=1, penalty='elasticnet', l1_ratio=0.3)
mdl.fit(Xtr, ytr)

print (mdl.score(Xtr, ytr))

result_col_2 = mdl.predict(get_test_data())
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('naive_linear_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction')
#print (mdl.score(Xts, yts))
# 0.845 best score on split data with a=0.00001, max_iter=1000, l1_ratio = 0.3
# .847 when trained on all data
