# bagged naive bayes
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from utils import *
import numpy as np

Xtr, ytr, Xts, yts = get_split_data()
#scale.fit(Xtr)
#Xtr = scale.transform(Xtr)
#Xts = scale.transform(Xts)
print ('loaded data')
mdl = BaggingClassifier(MultinomialNB(alpha=0),  verbose=1, n_jobs=-1, max_features=1.0)
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
