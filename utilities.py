import numpy as np
import matplotlib.pyplot as plt


# Custom Train-test split method
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    shuffled_indices = np.random.permutation(n_samples)
    test_set_size = int(n_samples*test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X.iloc[train_indices],X.iloc[test_indices],y.iloc[train_indices],y.iloc[test_indices]

    # Feature scaling
def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    # Use training mean & std to transform both train and test

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled, mean, std

class LogisticRegression:
    def __init__(self,learning_rate=0.001,lambda_reg=0.1,n_iters=1000):
        # Hyperparameters (chosen by user)
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.n_iters = n_iters

        # Model paramters (initialized by model)
        self.weights = None
        self.bias = None
        self.cost_hist = []
    
        # Activation function
    def _sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
        
        # Cost function
    def _compute_cost(self,y,y_hat):
        m = len(y)
        y_hat = np.clip(y_hat,1e-15,1-1e-15)
        term_1 = y*np.log(y_hat)
        term_2 = (1-y)*np.log(1-y_hat)
        cost = -np.sum(term_1+term_2)/m
        reg_cost = (self.lambda_reg/(2*m))*np.sum(np.square(self.weights))
        return cost + reg_cost
    
        # Computing the Gradient descnet
    def _compute_gradients(self,X,y,y_hat):
        m = len(y)
        # computing gradients (dw, db)
        dw = (X.T@(y_hat-y))/m + (self.lambda_reg/m)*self.weights
        db = (1/m)*np.sum(y_hat-y)

        # Updating the weights
        self.weights -= self.learning_rate*dw
        self.bias -= self.learning_rate*db

        # Model training
    def fit(self,X,y):
        m , n = X.shape
        self.weights = np.zeros(n) 
        self.bias = 0.0
        for i in range(self.n_iters):
            z = (X @ self.weights) + self.bias
            y_hat = self._sigmoid(z)
            self.cost_hist.append(self._compute_cost(y,y_hat))
            self._compute_gradients(X,y,y_hat)

        # Prediction
    def predict(self,X):
        z = X @ self.weights + self.bias
        y_hat = self._sigmoid(z)
        return np.where(y_hat>=0.5,1,0)
    
        # Plot of Cost function and number of iterations
    def plot_cost(self):
        plt.plot(self.cost_hist)
        plt.xlabel("Iteration")
        plt.ylabel("J(w,b)")
        plt.title("Cost Function ")
        plt.grid(True)
        plt.show()

