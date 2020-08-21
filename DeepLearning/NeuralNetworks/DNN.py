import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters_deep(layer_dims):
    np.random.seed(42)
    parameters = {}
    
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1] * 0.01)
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
        
    return AL, caches
    
def compute_cost(AL, Y):
    m = Y.shape[1]
    
    cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL))) / m
    
    cost = np.sequeeze(cost)
    
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid") 
    
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp    
        grads["db" + str(l + 1)] = db_temp
        
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters

def Visualize_LogLoss(epoch, costs):
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("epoch")
    plt.title("Learning rete = " + str(learning_rate))
    plt.grid()
    plt.show()

# pythonic
def L_layer_model(X, Y, layers_dims, learning_rate = 1e-4, iterations=3000, print_cost=False):
    np.random.seed(42)
    costs = []
    
    parameters = initialize_parameters_deep(layer_dims)
    
    for epoch in range(iterations):
        AL, caches = L_model_forward(X, parameters)
        
        cost = compute_cost(AL, Y)
        
        grads = L_model_backward(AL, Y, caches)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if epoch % 100 == 0:
            costs.append(cost)
            epoch.append(epoch)
        if print_logloss && epoch % 100 == 0:
            print("epoch {} , Logloss {} ".format(epoch, log_loss))
        
    Visualize_LogLoss(epoch, costs)