import numpy as np

class LogisticRegression:
    def __init__(self, data, learning_rate=0.01) -> None:
        self.data = data
        self.learning_rate = learning_rate
        self.weights = np.zeros(data.shape[1] - 1)
        self.bias = 0
        self.error_history = []

    def error(self, x: np.ndarray, y: np.ndarray) -> float:
        return -np.mean(y * np.log(self.predict(x)) + (1 - y) * np.log(1 - self.predict(x)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-np.dot(x, self.weights) - self.bias)))
    
    def __batch_gradient_descent(self, x: np.ndarray, y: np.ndarray, epochs) -> None:
        for _ in range(epochs):
            tmp_weights = self.learning_rate * np.dot(x.T, self.predict(x) - y) / y.size
            tmp_bias = self.learning_rate * np.sum(self.predict(x) - y) / y.size
            
            self.weights -= tmp_weights
            self.bias -= tmp_bias
            self.error_history.append(self.error(x, y))

    def __mini_batch_gradient_descent(self, x: np.ndarray, y: np.ndarray, epochs, batch_size) -> None:
        for _ in range(epochs):
            i = np.random.choice(x.shape[0], batch_size)
            s_x = x[i]
            s_y = y[i]
            tmp_weights = self.learning_rate * np.dot(s_x.T, self.predict(s_x) - s_y) / s_y.size
            tmp_bias = self.learning_rate * np.sum(self.predict(s_x) - s_y) / s_y.size
            
            self.weights -= tmp_weights
            self.bias -= tmp_bias
            self.error_history.append(self.error(x, y))

    def __stochastic_gradient_descent(self, x: np.ndarray, y: np.ndarray, epochs) -> None:
        for _ in range(epochs):
            i = np.random.choice(x.shape[0], 1)[0]
            s_x = x[i]
            s_y = y[i]
            tmp_weights = self.learning_rate * s_x * (self.predict(s_x) - s_y)
            tmp_bias = self.learning_rate * (self.predict(s_x) - s_y)
            
            self.weights -= tmp_weights
            self.bias -= tmp_bias
            self.error_history.append(self.error(x, y))

    def train(self, epochs=1000, mode=['batch']) -> None:
        x = self.data.iloc[:, 1:].values.astype(float)
        y = self.data.iloc[:, 0].values.astype(float)
        if mode[0] == 'mini-batch' and mode[1] < x.shape[0]:
            self.__mini_batch_gradient_descent(x, y, epochs, mode[1])
        elif mode[0] == 'stochastic':
            self.__stochastic_gradient_descent(x, y, epochs)
        else:
            self.__batch_gradient_descent(x, y, epochs)
