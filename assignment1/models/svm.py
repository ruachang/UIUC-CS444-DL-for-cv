"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, mini_batch:int , lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.mini_batch = mini_batch
    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        g = np.zeros((X_train.shape[0], self.w.shape[0], self.w.shape[1]))
        # STEP1: Predict y_hat
        
        # STEP2: Minus the value of the real class to get the loss of each class
        for i in range(len(X_train)):
            y_hat = np.dot(self.w.T, X_train[i])
            loss = y_hat[y_train[i]] - y_hat
            # STEP3: Update the weight using sign func(only update when > 0)
            for j in range(self.n_class):
                if loss[j] < 1:
                    if j != y_train[i]:
                        g[i, :, j] = X_train[i]
                        g[i, :, y_train[i]] -= X_train[i]
            g[i] = self.reg_const * self.w / self.n_class + g[i]
        g = np.mean(g, axis=0)
        return g

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # STEP0: Initialize w: W = class * features
        self.w = np.random.randn(X_train.shape[1], self.n_class)
        num_batch = int(len(X_train) / self.mini_batch)
        # print(X_train.shape, self.w.shape)
        for epoch in range(self.epochs):
            # Step0: shuffle the data every epoch and reduce lr depend on epoch
            index = np.arange(len(X_train))
            np.random.shuffle(index)
            X_train = X_train[index]
            y_train = y_train[index]
            if epoch == int(self.epochs * 0.9):
                self.lr = self.lr / 10
            # TODO: Use mini batch SGD instead of SGD
            for batch in range(num_batch):
                g = self.calc_gradient(X_train[batch : batch + self.mini_batch], y_train[batch : batch + self.mini_batch])
                self.w = self.w - self.lr * g
        return 

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        y_hat = np.dot(X_test, self.w)
        y_hat = np.argmax(y_hat, axis=1)
        return y_hat
