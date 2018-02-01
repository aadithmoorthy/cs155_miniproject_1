# utilities


import pandas
def get_split_data():
    data_tr = pandas.read_csv('data_tr.txt', delimiter=" ", header=None).as_matrix()
    data_val = pandas.read_csv('data_val.txt', delimiter=" ", header=None).as_matrix()

    Xtr = data_tr[:,1:]
    ytr = data_tr[:,0]
    Xts = data_val[:,1:]
    yts = data_val[:,0]

    return Xtr, ytr, Xts, yts

def get_unsplit_data():
    data = pandas.read_csv('training_data.txt', delimiter=" ").as_matrix()


    X = data[:,1:]
    y = data[:,0]


    return X,y

def get_test_data():
    data = pandas.read_csv('test_data.txt', delimiter=" ").as_matrix()

    return data
