"""Neural network model."""

from typing import Sequence

import numpy as np

from copy import deepcopy

class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

        # TODO: You may set parameters for Adam optimizer here
        self.m = 1
        self.v = 1
    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix(D, F)
            X: the input data(N, D)
            b: the bias(1, F)
        Returns:
            the output
        """
        # TODO: implement me
        return np.dot(X, W) + b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray,) -> np.ndarray:
        """Gradient of linear layer
            z = WX + b(zi = wi * x + bi)
            returns W, b, de_dx
        """
        # TODO: implement me
        # dz/dw = X, dz/db = 1, dz/dx = W
        return np.dot(de_dz, X).T, np.sum(de_dz, axis=1), np.dot(de_dz.T, W.T)

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return np.where(X > 0, 1, 0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return 1 / 2 * (1 + np.tanh(x / 2))
        
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return np.mean((y - p) ** 2)
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return -2 * (y - p) / len(y)
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return self.mse_grad(y, self.sigmoid(p)) * self.sigmoid_grad(p)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        self.outputs["0"] = deepcopy(X)
        for i in range(1, self.num_layers + 1):
            self.outputs[str(i)] = self.linear(self.params["W" + str(i)], X, self.params["b" + str(i)])
            X = self.relu(self.outputs[str(i)])
        self.outputs[str(self.num_layers + 1)] = self.sigmoid(self.outputs[str(self.num_layers)])
                                
        return self.outputs[str(self.num_layers + 1)]

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        #  STEP1 Calculate for the output and loss of the network
        loss = self.mse(y, self.outputs[str(self.num_layers + 1)])
        # STEP2 calculate the gradient of the last layer and the sigmoid
        self.gradients["W" + str(self.num_layers)], self.gradients["b" + str(self.num_layers)], \
        self.gradients["de_dx" + str(self.num_layers)] =  self.linear_grad(\
            self.params["W" + str(self.num_layers)], self.relu(self.outputs[str(self.num_layers - 1)]), self.params["b" + str(self.num_layers)], \
            (self.mse_sigmoid_grad(y, self.outputs[str(self.num_layers)])).T)
        
        for i in range(self.num_layers - 1, 1, -1):
            self.gradients["W" + str(i)], self.gradients["b" + str(i)],\
            self.gradients["de_dx" + str(i)] = self.linear_grad(\
                self.params["W" + str(i)], self.relu(self.outputs[str(i - 1)]), self.params["W" + str(i)], \
                self.gradients["de_dx" + str(i + 1)].T * self.relu_grad(self.outputs[str(i)]).T)
            
        self.gradients["W" + str(1)], self.gradients["b" + str(1)],\
        self.gradients["de_dx" + str(1)] = self.linear_grad(\
            self.params["W" + str(1)], (self.outputs[str(0)]), self.params["b" + str(self.num_layers)], \
            self.gradients["de_dx" + str(2)].T * self.relu_grad(self.outputs[str(1)]).T)
        return loss * y.shape[1]

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
            for i in range(1, self.num_layers + 1):
                # print(self.params["W" + str(i)], params["b" + str(i)])
                self.params["W" + str(i)] += -lr * self.gradients["W" + str(i)]
                self.params["b" + str(i)] += -lr * self.gradients["W" + str(i)]
                # print(self.params["W" + str(i)], params["b" + str(i)])
        else:
            for i in range(1, self.num_layers + 1):
                self.m[i] = b1 * self.m[i] + (1 - b1) * self.gradients["W" + str(i)]
                self.v[i] = b2 * self.v[i] + (1 - b2) * self.gradients["W" + str(i)] ** 2
                self.params["W" + str(i)] += -lr * self.m[i] / (np.sqrt(self.v[i]) + eps)
                self.params["b" + str(i)] += -lr * self.m[i] / (np.sqrt(self.v[i]) + eps)
        return