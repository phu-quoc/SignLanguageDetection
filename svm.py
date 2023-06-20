import numpy as np

class SVM:
    def __init__(self, num_classes, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.classes = np.unique(y)
        self.weights = np.zeros((self.num_classes, num_features))
        self.bias = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            y_binary = np.where(y == self.classes[i], 1, -1)
            weights = np.zeros(num_features)
            bias = 0
            
            for _ in range(self.num_iterations):
                for j in range(num_samples):
                    # y_binary = 1, Condition >= 1 có nghĩa là nó là thuộc vào loại đó
                    
                    condition = y_binary[j] * (np.dot(X[j], weights) - bias) >= 1
                    gradient_weights = self.lambda_param * weights - (1 / num_samples) * y_binary[j] * X[j]
                    gradient_bias = - (1 / num_samples) * y_binary[j]
                    
                    # Cập nhật lại trọng số và bias
                    if condition:
                        weights -= self.learning_rate * gradient_weights
                        bias -= self.learning_rate * gradient_bias
                    else:
                        weights -= self.learning_rate * (gradient_weights - self.lambda_param * weights)
                        bias -= self.learning_rate * gradient_bias
            
            # gán trọng số của và bias
            self.weights[i] = weights
            self.bias[i] = bias
        return self
    
    def predict(self, X):
        scores = np.dot(X, self.weights.T) - self.bias
        index = np.argmax(scores, axis=1)
        predictions = self.classes[index]
        return predictions