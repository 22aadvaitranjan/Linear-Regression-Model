import numpy as np

#Model
class LinearRegression:
    def __init__(self, learing_rate, epochas):
        self.learning_rate = learing_rate
        self.epochas = epochas
        
        self.weight = None
        self.bias = None
        
        self.loss_history = np.array([])
        
    def fitness(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        self.weight = np.zeros(n_features)
        self.bias = 0.0
        
        for epoch in range(self.epochas):
            
            predictions = X@self.weight + self.bias
            
            error = predictions-y
            
            Loss = np.mean(error**2)
            self.loss_history = np.append(self.loss_history, Loss)
            
            dw = 2/n_samples * (X.T@error)
            db = 2/n_samples * np.sum(error)
            
            self.weight -= self.learning_rate*dw
            self.bias -= self.learning_rate*db
            
            if epoch % 100 == 0:
                print(
                    f"Epoch: {epoch:4d}  "
                    f"Loss: {Loss:.6f}"
                )
    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.weight + self.bias

    
    
    
#Put your desired values here
    
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

#Put your training values here

y = np.array([
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16
])

model = LinearRegression(learing_rate=0.03, epochas=2000)

model.fitness(X, y)


#Put your testing data here
test_data = np.array([
    [9],
    [10],
    [20]
])


predictions = model.predict(test_data)

print("\nPredictions:")

for x, prediction in zip(test_data, predictions):
    print(f"x = {x[0]:2d} -> y = {prediction:.4f}")
    