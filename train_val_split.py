# repeatable train validation split for data; at initial stage, need a split
# so that we can avoid overfitting and train some hyperparameters

from sklearn.model_selection import train_test_split
import time
import numpy as np
t = time.time()

data = np.loadtxt('training_data.txt', skiprows=1)
print(data.shape)
print('took', time.time()-t)

X = data[:,1:]
y = data[:,0]

data_tr, data_val = train_test_split(data, test_size=0.05, random_state = 42)

np.savetxt('data_val.txt', data_val, fmt="%d")

np.savetxt('data_tr.txt', data_tr, fmt="%d")
