import numpy as np
import matplotlib.pyplot as plt


def Layer_Size(X, y):
    X_dims = X.shape[0]
    hidden_neurons = 3
    y_dims = y.shape[0]
    return (X_dims, hidden_neurons, y_dims)

def Initialize(X_dims, hidden_neurons, y_dims):
    W1 = np.random.randn(hidden_neurons, X_dims) * 0.01
    b1 = np.zeros((hidden_neurons, 1))
    
    W2 = np.random.randn(y_dims, hidden_neurons) * 0.01
    b2 = np.zeros((y_dims, 1))
    
    parameters = {"W1" : W1, "b1" : b1,
                  "W2" : W2, "b2" : b2}
    
    return parameters

def Feed_Forward_Propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, x) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    output = sigmoid(Z2)
    
    cache = {"Z1" : Z1, "A1" : A1,
             "Z2" : Z2, "output" : output}
    
    return output, cache

def Compute_Logloss(output, y, parameters):

    log_probs = np.multiply(np.log(output), y) + np.multiply(np.log(1-output), 1-y)
    
    cost = -1/y.shape[1] * np.sum(logprobs)
    
    cost = np.sequeeze(cost)
    
    return cost

def Backward_Propagate(parameters, cache, X, y):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    output = cache["output"]
    
    dZ2 = output - y
    dW2 = 1/X.shape[1] * np.dot(dZ2, A1.T)
    db2 = 1/X.shape[1] * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1/X.shape[1] * np.dot(dZ1, X.T)
    db1 = 1/X.shape[1] * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1" : dW1, "db1" : db1
             "dW2" : dW2, "db2" : db2}
    
    return grads

def Update_Parameters(parameters, grads, learning_rate = 1e-2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 -= dW1 * learning_rate
    b1 -= db1 * learning_rete
    W2 -= dW2 * learning_rate
    b2 -= db2 * learning_rete
    
    parameters = {"W1" : W1, "b1" : b1,
                  "W2" : W2, "b2" : b2}
    
    return parameters

def Visualize_LogLoss(epoch, logloss):
    plt.plot(epoch, logloss)
    plt.xlabel("epoch")
    plt.ylabel("logloss")
    plt.title("LogLoss for training")
    plt.grid()
    plt.show()


# pythonic
def Perceptron_Model(X, y, iterations=1000, learning_rate=1e-2, print_logloss=False):
    np.random.seed(42)
    logloss = []
    epoch = []
    
    X_dims, hidden_neurons, y_dims = Layer_Size(X, y)
    
    parameters = Initialize(X_dims, hidden_neurons, y_dims)
    
    for epoch in range(iterations):
        output, cache = Feed_Forward_Propagation(X, parameters)
        
        cost = Compute_Logloss(output, y, parameters)
        
        grads = Backward_Propagate(parameters, cache, X, y)
        
        parameters = Update_Parameters(parameters, grads, learning_rate = 1e-2)
        
        if epoch % 100 == 0:
            logloss.append(log_loss)
            epoch.append(epoch)
        if print_logloss && epoch % 100 == 0:
            print("epoch {} , Logloss {} ".format(epoch, log_loss))
        
    Visualize_LogLoss(epoch, logloss)