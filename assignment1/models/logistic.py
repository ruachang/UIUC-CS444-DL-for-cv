"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.stop_threshold = 1e-6

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: sigmoid(z) = 1 / (1 + exp(-z))
        if z > 0:
            return 1 / (1 + np.exp(-z))
        else:
            return (1 - 1 / (1 + np.exp(z)))
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # STEP0: Change y_train to -1/1 and initialize w 
        self.w = np.random.randn(X_train.shape[1])
        y_train = y_train * 2 - 1
        for epoch in range(self.epochs):
            # Step0: shuffle the data every epoch and reduce lr depend on epoch
            index = np.arange(len(X_train))
            np.random.shuffle(index)
            X_train = X_train[index]
            y_train = y_train[index]
            if epoch == int(self.epochs * 0.9):
                self.lr = self.lr / 10
            # Use SGD
            for i in range(len(X_train)):
                xi, yi = X_train[i], y_train[i]
                # STEP1: pridict current y_hat with current weight w
                y_hat = np.dot(self.w.T, xi)
                # STEP2: with initialized weight w, calculate the loss 
                loss = self.sigmoid(-yi * y_hat) * yi * xi
                # print("yi", yi.shape, "y_hat", y_hat.shape, "xi", xi.shape, "loss", loss.shape)
                # STEP3: Based on loss, determine whether to update the weight w or stop updating
                if np.linalg.norm(loss) > self.stop_threshold:
                    self.w = self.w + self.lr * loss
                else: 
                    continue
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
        # STEP1: pridict current y_hat with current weight w
        y_hat = np.dot(X_test, self.w)
        # Step2: change y_hat to 0/1
        for i in range(len(y_hat)):
            if y_hat[i] > self.threshold:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        return y_hat
