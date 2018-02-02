# platt scaling
from sklearn.linear_model import LogisticRegression

def probabilities(Xtr, ytr, Xts):
    model = LogisticRegression()
    model.fit(Xtr,ytr)
    print 'platt scaling accuracy', model.score(Xtr,ytr)
    return model.predict_proba(Xtr), model.predict_proba(Xts)
