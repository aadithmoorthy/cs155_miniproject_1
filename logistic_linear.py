# logistic linear
# using standard scaler helps!

from sklearn.linear_model import LogisticRegression
from utils import *
import numpy as np
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
Xtr, ytr, Xts, yts = get_split_data()
Xtr = np.concatenate((Xtr, np.power(Xtr,2), np.log(Xtr+.00000001),1/(Xtr+.1)), axis=1)
Xts = np.concatenate((Xts, np.power(Xts,2), np.log(Xts+.00000001), 1/(Xts+.1)), axis=1)

scale.fit(Xtr)
Xtr = scale.transform(Xtr)
Xts = scale.transform(Xts)
print ('loaded data')
mdl = LogisticRegression(C=0.00008)
mdl.fit(Xtr, ytr)

print (mdl.score(Xtr, ytr))
print (mdl.score(Xts, yts))
#print (mdl.predict(Xts))
#print (mdl.predict_proba(Xts))

# best on split - 0.853 with c = 0.0005 +scaler
# best on unsplit - 0.854 with c = 0.0005 +scaler
'''
result_col_2 = mdl.predict(scale.transform(get_test_data()))
result_col_2 = result_col_2.reshape((len(result_col_2),1))
result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
result = np.concatenate((result_col_1,result_col_2), axis = 1)
np.savetxt('nonlinear_log_reg_result.txt', result, fmt="%d", delimiter=',', header='Id,Prediction')
'''
#result = mdl.predict_proba(scale.transform(get_test_data()))[:,1]
#np.savetxt('logistic_linear_result_probabilities.txt', result, fmt="%g", delimiter=',')

'''
result = mdl.predict_proba(Xts)[:,1]

np.savetxt('logistic_linear_merge_result.txt', result, fmt="%g", delimiter=',')'''
#print (mdl.score(Xts, yts))'''
