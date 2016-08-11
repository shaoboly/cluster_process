import numpy as np

ma = np.ones((3, 2))
#print ma

add = np.array([0,1])
data = np.vstack((ma,add))
#print ma

list = [[0]*10]*10
#list = [[0 for y in range(5)] for x in range(10)]
list[0][0] = 1

print list