import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.random.randn(input_size + 1) * 0.01  # Inclui o termo de bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        """Step function as the activation function."""

        ### START CODE HERE ###
        return np.where(x >= 0, 1, -1)
        ### END CODE HERE ###
    
    def predict(self, X):
        """Realize the prediction of the perceptron for the input data X."""

        ### START CODE HERE ###
        X_with_bias = np.c_[X, np.ones(X.shape[0])]  # Adiciona bias
        return np.array([self.activation(np.dot(x, self.weights)) for x in X_with_bias])
        ### END CODE HERE ###


    def fit(self, X, y):
        """Fit the perceptron model to the training data (X, y)."""

        ### START CODE HERE ###
        X_with_bias = np.c_[X, np.ones(X.shape[0])]  # Adiciona bias
        for _ in range(self.epochs):  # Itera sobre as épocas
            for i in range(X.shape[0]):  # Itera sobre os dados de treinamento
                y_pred = self.activation(np.dot(X_with_bias[i], self.weights))  # Previsão
                error = y[i] - y_pred  # Calcula o erro
                self.weights += self.learning_rate * error * X_with_bias[i]
        ### END CODE HERE ###

def generate_data(seed, samples, noise):
    """
    Generates two-class classification data with overlapping clusters.
    """
    np.random.seed(seed)
    X, y = make_blobs(n_samples=samples, centers=2, cluster_std=noise, random_state=seed)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for perceptron
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train a single-layer perceptron using NumPy.")
    parser.add_argument('--registration_number', type=int, required=True, help="Student's registration number (used as seed)")
    parser.add_argument('--samples', type=int, default=200, help="Number of data samples")
    parser.add_argument('--noise', type=float, default=1.5, help="Standard deviation of clusters")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Perceptron learning rate")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs")
    
    args = parser.parse_args()

    # Generate data
    X, y = generate_data(args.registration_number, args.samples, args.noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.registration_number)

    # Train the perceptron
    perceptron = Perceptron(input_size=2, learning_rate=args.learning_rate, epochs=args.epochs)
    perceptron.fit(X_train, y_train)

    # Evaluate the perceptron
    y_pred = perceptron.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"Perceptron Training Completed.")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Visualizing decision boundary
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k", alpha=0.6)
    
    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid_points).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.title(f"Perceptron Decision Boundary (Accuracy: {accuracy:.4f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == "__main__":
    main()