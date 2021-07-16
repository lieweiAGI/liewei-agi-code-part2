import numpy as np

a = np.array([8,8,0,1])

def one_hot(x):
    z = np.zeros(shape=[4,10])
    for i in range(4):
        index = int(x[i])
        z[i][index] = 1
    return z

print(one_hot(a))