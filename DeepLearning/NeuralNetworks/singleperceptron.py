import numpy as np
import matplotlib.pyplot as plt


def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def Initialize(dims):
    W = np.zeros((dims, 1))
    b = 0.0
    return W, b

def Forward_Propagate(W, b, X, y):
    y_hat = Sigmoid(np.dot(W.T, X) + b)
    log_loss = -1 / X.shape[1] * np.sum( y*np.log(y_hat) + (1-y)*np.log(1-y_hat) )
    
    dW = np.dot(X, (y_hat - y).T) / X.shape[1]
    db = np.sum(y_hat - y) / X.shape[1]
    
    grads = { "dW" : dW, "db" : db}
    
    return grads, log_loss

def Back_Propagate(W, b, X, y, iterations, learning_rate, print_logloss=False):
    logloss = []
    epoch = []
    for epoch in range(iterations):
        grads, log_loss = Forward_Propagate(W, b, X, y)
        dW = grads['dW']
        db = grads['db']
        
        W -= learning_rate * dW
        b -= learning_rate * db
        
        if epoch % 100 == 0:
            logloss.append(log_loss)
            epoch.append(epoch)
        if print_logloss && epoch % 100 == 0:
            print("epoch {} , Logloss {} ".format(epoch, log_loss))
            
    grads = { "dW" : dW, "db" : db}
    params = { "W" : W, "b" : b}
    
    return params, grads, logloss, epoch

def Visualize_LogLoss(epoch, logloss):
    plt.plot(epoch, logloss)
    plt.xlabel("epoch")
    plt.ylabel("logloss")
    plt.title("LogLoss for training")
    plt.grid()
    plt.show()

def Predict_Accuracy(W, b, X, y):
    y_pred = np.zeros((1, X.shape[1]))
    
    W = W.reshape(x.shape[0], 1)
    probability = Sigmoid(np.dot(W.T, W) + b)
    for index in range(probability.shape[1]):
        if probability[:, index] > 0.5:
            y_pred[:, index] = 1
        else:
            y_pred[:, index] = 0
    
    accuracy = (1 - np.mean(np.abs(y_pred - y))) * 100
    
    return accuracy


# pythonic
def Model(X_train, y_train, X_test, y_test, iterations=1000, learning_rate=1e-2, print_logloss=False):
    W, b = Initialize(X_train.shape[0])
    
    params, _, logloss, epoch = Back_Propagate(W, b, X_train, y_train, iterations, learning_rate, print_logloss)
    
    W = params["W"]
    b = params["b"]
    
    print("Train accuracy: ", Predict_Accuracy(W, b, X_train, y_train))
    print("Test accuracy: ", Predict_Accuracy(W, b, X_test, y_train))
    
    Visualize_LogLoss(epoch, logloss)