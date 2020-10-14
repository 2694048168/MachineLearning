import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def initialize(x):
	w = np.zeros(x.shape[1], 1)
	b = 0.0
	return w, b

def connection(x, w, b):
	z = np.dot(w, x) + b
	return z

def FeedForwardPropagate(x, y, w, b):
	y_hat = sigmoid(connection(x, w, b))

	logloss = -1/x.shape[1] * np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

	dw = np.dot(x, (y_hat - y)) / x.shape[1]
	db = np.sum(y_hat - y) / x.shape[1]

	return logloss, dw, db

def BackPropagate(w, b, x, y, iterations, learning_rate, loss):
	logloss = []
	epoch = []
	for epoch in range(iterations):
		logloss, dw, db = FeedForwardPropagate(x, y, w, b)

		w = w - learning_rate * dw
		b = b - learning_rate * db

		if epoch % 100 == 0:
			epoch.append(epoch)
			logloss.append(logloss)

		if logloss < loss:
			break

	return epoch, logloss, w, b

def predict(w, b, test, y_test):
	y_pred = sigmoid(connection(test, w, b))

	for index in range(y_pred):
		if test[:, index] > 0.5:
			test[:, index] = 1
		else:
			test[:, index] = 0

    corrects = (y_pred == y_test)++
	accuracy = corrects / test.shape[1]

	return accuracy

def Visualize_logloss(epoch, logloss):
	plt.plot(epoch, logloss)
	plt.xlabel("epoch")
	plt.ylabel("logloss")
	plt.title("LogLoss for training")
	plt.show()

def processing():
	df = pd.read_csv("./dataset/breast_cancer.csv")

	return x, y, test, y_test

def main():
	x, y = processing()

	w, b = initialize(x)

	logloss, dw, db = FeedForwardPropagate(x, y, w, b)

	epoch, logloss, w, b = BackPropagate(dw, db, x, y, 1000, 1e-2, 1e-5)

	Visualize_logloss(epoch, logloss)

	print(predict(w, b, test))


if __name__ == '__main__':
	main()
	