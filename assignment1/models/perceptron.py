"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # STEP0: Initialize w: W = class * features
        np.random.seed(1)
        # X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / np.std(X_train, axis=1, keepdims=True) 
        self.w = np.random.randn(X_train.shape[1], self.n_class)
        # print(X_train.shape, self.w.shape)
        for epoch in range(self.epochs):
            # Step0: shuffle the data every epoch and reduce lr depend on epoch
            index = np.arange(len(X_train))
            np.random.shuffle(index)
            X_train = X_train[index]
            y_train = y_train[index]
            if epoch == int(self.epochs * 0.8):
                self.lr = self.lr / 10
            # Use SGD
            for i in range(len(X_train)):
                # STEP1: Predict y_hat
                y_hat = np.dot(self.w.T, X_train[i])
                # STEP2: Minus the value of the real class to get the loss of each class
                loss = y_hat - y_hat[y_train[i]]
                # STEP3: Update the weight using sign func(only update when > 0)
                for j in range(self.n_class):
                    if loss[j] > 0:
                        if j != y_train[i]:
                            self.w[:, j] = self.w[:, j] - self.lr * X_train[i]
                            self.w[:, y_train[i]] = self.w[:, y_train[i]] + self.lr * X_train[i]
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
        y_hat = np.dot(X_test, self.w)
        y_hat = np.argmax(y_hat, axis=1)
        
        return y_hat
