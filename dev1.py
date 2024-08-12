import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def load_data(self, file_path):
        # Implement data loading from CSV
        return np.genfromtxt(file_path, delimiter=',', skip_header=1)

    def shuffle_data(self, data):
        np.random.shuffle(data)

    def sub_array(self, array, start, end):
        return array[start:end]

    def transpose(self, matrix):
        return np.transpose(matrix)

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss_value = self.loss.forward(y_pred, y)
            grad_output = self.loss.backward(y_pred, y)
            self.backward(grad_output)
            self.optimizer.step(self.layers)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_value:.4f}')

    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        loss_value = self.loss.forward(y_pred, y)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
        print(f'Loss: {loss_value:.4f}, Accuracy: {accuracy:.4f}')

class Linear:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, X):
        self.input = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, grad_output):
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.weights.T)

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, X):
        self.input = X
        return np.maximum(0, X)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, X):
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        return grad_output  # Simplified for this implementation

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-12, 1. - 1e-12)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        return -np.mean(np.log(correct_confidences))

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        grad = y_pred
        grad[range(samples), y_true] -= 1
        return grad / samples

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.bias -= self.learning_rate * layer.grad_bias

# Main execution
if __name__ == '__main__':
    # Load and preprocess data
    nn = NeuralNetwork()
    data = nn.load_data('/path/to/train.csv')
    m, n = data.shape

    # Shuffle data
    nn.shuffle_data(data)

    # Prepare development and training data
    data_dev = nn.transpose(nn.sub_array(data, 0, 1000))
    Y_dev = data_dev[0]
    X_dev = nn.transpose(nn.sub_array(data_dev, 1, len(data_dev))) / 255.0

    data_train = nn.transpose(nn.sub_array(data, 1000, m))
    y_train = data_train[0]
    x_train = nn.transpose(nn.sub_array(data_train, 1, len(data_train))) / 255.0

    # Instantiate and build the model
    model = NeuralNetwork()
    model.add_layer(Linear(784, 128))
    model.add_layer(ReLU())
    model.add_layer(Linear(128, 10))
    model.add_layer(Softmax())

    # Compile the model with loss and optimizer
    loss = CrossEntropyLoss()
    optimizer = SGD(learning_rate=0.2)
    model.compile(loss, optimizer)

    # Train the model
    model.train(x_train, y_train, epochs=150)

    # Evaluate the model
    model.evaluate(X_dev, Y_dev)
