import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LogisticNeuron:
    def __init__(self, input_dim, learning_rate=0.1, epochs=1000):
        ### START CODE HERE ###
        
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn(1)
        self.loss_history = []

        ### END CODE HERE ###
    
    def sigmoid(self, z):
        ### START CODE HERE ###
        
        return 1 / (1 + np.exp(-z))

        ### END CODE HERE ###
    
    def predict_proba(self, X):
        ### START CODE HERE ###

        z = (X @ self.weights) + self.bias
        return self.sigmoid(z)

        ### END CODE HERE ###
    
    def predict(self, X):
        ### START CODE HERE ###

        return (self.predict_proba(X) > 0.5).astype(int)

        ### END CODE HERE ###
    
    def train(self, X, y):
        ### START CODE HERE ###

        for epoch in range(self.epochs):            
            y_pred = self.predict_proba(X) 
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.loss_history.append(loss)

            gradient_w = (X.T @ (y_pred - y)) / len(y)
            gradient_b = np.mean(y_pred - y)
            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

        ### END CODE HERE ###


def generate_dataset():
    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2.0)
    return X, y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Logistic Regression Output')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

def plot_loss(model):
    plt.plot(model.loss_history, 'k.')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over Training Iterations')
    plt.show()

# Generate dataset
X, y = generate_dataset()

# Train the model
neuron = LogisticNeuron(input_dim=2, learning_rate=0.1, epochs=100)
neuron.train(X, y)

# Plot decision boundary
plot_decision_boundary(neuron, X, y)

# Plot loss over training iterations
plot_loss(neuron)