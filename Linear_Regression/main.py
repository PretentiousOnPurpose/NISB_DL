import numpy as np
import pandas as pd

ds = pd.read_csv('USA_Housing.csv')
X = np.array(ds.iloc[: , :-1])
y = np.array(ds.iloc[: , -2:-1])

w = np.random.rand(X.shape[1]+1)

from sklearn.preprocessing import MinMaxScaler
MMSx = MinMaxScaler()
MMSy = MinMaxScaler()
MMSx.fit(X)
MMSy.fit(y)
X = MMSx.transform(X)
X = np.append(arr=np.ones((len(X), 1)), values=X, axis=1)
y = MMSy.transform(y)

from sklearn.model_selection import train_test_split
xtr, xts,  ytr, yts = train_test_split(X, y, test_size=0.001)

def train(xtr, ytr, steps):
    for s in range(steps):
        for i in range(len(xtr)):
            y = np.array(xtr[i]).dot(w)
            err = ytr[i] - y
            backprop(err, xtr[i], 0.05)
            
def test(xts):
    res = []
    for i in xts:
        res.append(Predict(i))
    return res
            
def Predict(InV):
    return MMSy.inverse_transform(np.array(InV).dot(w)) 

def backprop(error, inp, rate):
    for i in range(len(w)):
        w[i] = w[i] + rate*error*inp[i]


train(xtr, ytr,1)
res = test(xts)

