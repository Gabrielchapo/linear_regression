import numpy as np
from LinearRegression import LinearRegression

# EXTRACT DATA FROM FILE AND CHANGE IT TO THE RIGHT FORMAT
matrix = np.loadtxt("text", delimiter=',', skiprows=1).T
X, Y = matrix[0], matrix[1]
m = X.shape[0]
X = X.reshape((m,1))
Y = Y.reshape((m,1))

model = LinearRegression()

model.fit(X,Y, visualize=True)

model.save()
