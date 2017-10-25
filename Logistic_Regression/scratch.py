import numpy as np
import pandas as pd
import bigfloat

ds = pd.read_csv('data.csv')
y_temp = ds.iloc[: , 1:2].values
y = []
for i in y_temp:
    if i == 'M':
        y.append(1)
    elif i == 'B':
        y.append(0)
        
x = list(ds.iloc[: , 2:32].values)
x = np.append(arr=np.ones((569, 1)), values=x, axis=1)
w = np.random.randn(31)

from sklearn.model_selection import train_test_split
xtr, xts,  ytr, yts = train_test_split(x, y, test_size=0.2)

def train(xtr, ytr, steps):
    for s in range(steps):
        for i in range(len(xtr)):
            y = np.array(xtr[i]).dot(w)
            err = ytr[i] - sigmoid(y)
            backprop(err, xtr[i], 0.005)

def test(xts, yts):
    acc = 0
    for i in range(len(xts)):
        y = np.array(xts[i]).dot(w)
        if sigmoid(y) == yts[i]:
            acc += 1
    return acc/len(xts)

def backprop(error, inp, rate):
    for i in range(len(w)):
        w[i] = w[i] + rate*error*inp[i]
        #print(w[i])
        
def sigmoid(X):
    return 1/(1+bigfloat.exp(-X))

train(xtr, ytr,10)
test(xts , yts)
