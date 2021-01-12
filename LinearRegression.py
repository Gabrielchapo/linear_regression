import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:


    def fit(self, X, Y, gradient=0.95, nb_epochs=200, visualize=False):

        # number of observations
        m = X.shape[0]

        # standardization
        self.mu = sum(X) / m
        self.sigma = np.amax(X) - np.amin(X)
        X_train = (X - self.mu) / self.sigma

        # adding the biais to the dataset
        X_train = np.insert(X_train, X_train.shape[1], 1, axis=1)

        # init the weights
        self.thetas = np.zeros((X_train.shape[1], 1))

        for i in range(nb_epochs):

            predicted = np.dot(X_train, self.thetas)
            diff = predicted - Y
            gradient_vector = np.dot(X_train.T, diff)
            self.thetas -= (gradient / m) * gradient_vector

            loss = np.sum((diff ** 2)) / (2 * m)
            print("Epoch:", i, "=== Loss:", loss)


        # Normal Equation
        step1 = np.dot(X_train.T, X_train)
        step2 = np.linalg.pinv(step1)
        step3 = np.dot(step2, X_train.T)
        thetas = np.dot(step3, Y)

        if visualize is True:
            # PLOT IT
            plt.title('ft_linear_regression')

            y = np.dot(X_train, self.thetas)
            plt.plot(X, y, '-r', label='Gradient Descent')

            y = np.dot(X_train, thetas)
            plt.plot(X, y, '-b', label='Normal Equation')

            plt.plot(X, Y, 'gs', label="distribution")
            plt.xlabel('mileage', color='#1C2833')
            plt.ylabel('price', color='#1C2833')
            plt.legend(loc='upper right')
            plt.show()

    def save(self):
        dict = {"thetas":self.thetas, "mu":self.mu, "sigma":self.sigma}
        np.save("weights.npy", dict)

    def load(self):
        dict = np.load("weights.npy", allow_pickle='TRUE').item()
        self.sigma = dict["sigma"]
        self.mu = dict["mu"]
        self.thetas = dict["thetas"]

    def predict(self, X):
        X = (X - self.mu) / self.sigma
        prediction = X * self.thetas[0] + self.thetas[1]
        return round(prediction[0])