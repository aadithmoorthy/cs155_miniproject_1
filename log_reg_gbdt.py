# log_reg followed by gbdt

from sklearn.linear_model import LogisticRegression,LinearRegression, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from utils import *
import numpy as np

Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')
mdl = LinearRegression()
mdl.fit(Xtr, ytr)

print (accuracy_score(ytr, np.minimum(np.maximum(np.round(mdl.predict(Xtr)),0),1)))
print (accuracy_score(yts, np.minimum(np.maximum(np.round(mdl.predict(Xts)),0),1)))

ytr_resid = ytr - mdl.predict(Xtr)

yts_resid = yts - mdl.predict(Xts)

mdl_resid = GradientBoostingRegressor()
print(mdl_resid)
mdl_resid.fit(Xtr, ytr_resid)
print (mdl_resid.score(Xtr, ytr_resid))
print (mdl_resid.score(Xts, yts_resid))
preds = np.minimum(np.maximum(np.round(mdl.predict(Xts)+mdl_resid.predict(Xts)),0),1)
print (accuracy_score(yts, preds))

predtr = np.minimum(np.maximum(np.round(mdl.predict(Xtr)+mdl_resid.predict(Xtr)),0),1)
print (accuracy_score(ytr, predtr))
