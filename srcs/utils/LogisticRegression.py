import numpy as np

class LogisticRegression:
    def __init__(self, data, learning_rate=0.01) -> None:
        self.data = data
        self.learning_rate = learning_rate
        self.weights = np.zeros(data.shape[1] - 1)
        self.bias = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-np.dot(x, self.weights) - self.bias)))
    
    def train(self, epochs=1000) -> None:
        x = self.data.iloc[:, 1:].values.astype(float)
        y = self.data.iloc[:, 0].values.astype(float)
        for _ in range(epochs):
            tmp_weights = self.learning_rate * np.dot(x.T, self.predict(x) - y) / y.size
            tmp_bias = self.learning_rate * np.sum(self.predict(x) - y) / y.size
            
            self.weights -= tmp_weights
            self.bias -= tmp_bias
