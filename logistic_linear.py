# logistic linear

from sklearn.linear_model import LogisticRegression
from utils import *
import numpy as np

Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')
mdl = LogisticRegression(C=0.09)
mdl.fit(Xtr, ytr)

print (mdl.score(Xtr, ytr))
print (mdl.score(Xts, yts))
#print (mdl.predict(Xts))
#print (mdl.predict_proba(Xts))

# best on split - 0.845 with c = 0.09
# best on unsplit - 0.851 with c = 0.09

result_col_2 = mdl.predict(get_test_data())
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('logistic_linear_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction')

result = mdl.predict_proba(get_test_data())[:,1]
np.savetxt('logistic_linear_result_probabilities.txt', result, fmt="%g", delimiter=',')


result = mdl.predict_proba(Xts)[:,1]

np.savetxt('logistic_linear_merge_result.txt', result, fmt="%g", delimiter=',')
#print (mdl.score(Xts, yts))
