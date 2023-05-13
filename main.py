import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN_UNIT = 128
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [ NUM_HIDDEN_UNIT ]
NUM_OUTPUT = 10

l2_alpha = 1e-6
epochs = 100
batch_size = 64
lr = 0.01

def unpack (weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN[0]
    W = weights[start:end]
    Ws.append(W)

    # Unpack the weight matrices as vectors
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i]*NUM_HIDDEN[i+1]
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN[-1]*NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    # Reshape the weight "vectors" into proper matrices
    Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i-1])
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i+1]
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs

def softmax(z):
  y_hat = np.exp(z) / np.sum(np.exp(z), axis  = 0, keepdims = True)
  return y_hat

def accuracy(y_hat, Y):
  y_hat = np.argmax(y_hat, axis=0)
  Y_one_hot = np.eye(10)[Y.flatten()].T
  Y_one_hot = np.argmax(Y_one_hot, axis=0)
  return np.mean(y_hat == Y_one_hot)

def fCE (X, Y, weights):
    Ws, bs = unpack(weights)
    
    Z = []
    H = []
    
    z_temp = Ws[0].dot(X) + (bs[0].reshape(NUM_HIDDEN_UNIT, 1))
    Z.append(z_temp)
    
    h_temp = np.maximum(z_temp, 0)
    H.append(h_temp)
    
    for i in range(1, NUM_HIDDEN_LAYERS):
      z_temp = Ws[i].dot(h_temp) + bs[i].reshape(NUM_HIDDEN_UNIT,1)
      Z.append(z_temp)
      h_temp = np.maximum(z_temp, 0)
      H.append(h_temp)
      
    z_temp = Ws[-1].dot(H[-1]) + (bs[-1].reshape(NUM_OUTPUT, 1))
    Z.append(z_temp)
    y_hat = softmax(z_temp)
    
    Y_one_hot = np.eye(10)[Y.flatten()].T
    ce = -np.mean(np.sum(Y_one_hot * np.log(y_hat), axis=0))
    
    return ce, Z, H, y_hat

def gradCE (X, Y, weights):
  
  cross_entropy_loss, Z, H, y_hat = fCE(X, Y, weights)
  
  Ws, bs = unpack(weights)
  
  gradients_W = []
  gradients_bias = []
  n = y_hat.shape[1]
  
  Y_one_hot = np.eye(10)[Y.flatten()].T
  g = (y_hat - Y_one_hot) / n
  
  for i in range(NUM_HIDDEN_LAYERS, -1, -1):
    gradients_bias.append(np.sum(g, axis = 1))
    
    if i == 0:
      gradients_W.append((g.dot(X.T)) + (l2_alpha * Ws[i]))
      break

    else:
      gradients_W.append((g.dot(H[i-1].T)) + (l2_alpha * Ws[i]))

    g = Ws[i].T.dot(g)
    g = g * (Z[i-1] > 0)

  gradients_W.reverse()
  gradients_bias.reverse()

  allGradientsAsVector = np.hstack([ W.flatten() for W in gradients_W ] + [ b.flatten() for b in gradients_bias])

  return allGradientsAsVector

def show_W0 (W):
    Ws,bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()

def initWeightsAndBiases ():
    Ws = []
    bs = []

    # Strategy:
    # Sample each weight from a 0-mean Gaussian with std.dev. of 1/sqrt(numInputs).
    # Initialize biases to small positive number (0.01).

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN[i], NUM_HIDDEN[i+1]))/NUM_HIDDEN[i]**0.5) - 1./NUM_HIDDEN[i]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i+1])
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1]))/NUM_HIDDEN[-1]**0.5) - 1./NUM_HIDDEN[-1]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs

def train (trainX, trainY, weights, testX, testY, lr = 5e-2):

    for i in range(epochs):
        start = 0

        for j in range(int(trainX.shape[1] / batch_size)):
            end = start + batch_size

            X_batch = trainX[:, start:end]
            Y_batch = trainY[:, start:end]

            start = start + batch_size

            gradients = gradCE(X_batch, Y_batch, weights)

            weights = weights - lr * gradients

        cross_entropy_loss, Z, H, y_hat = fCE(testX, testY, weights)
        acc = accuracy(y_hat, testY)
        print("Epoch: "+str(i+1)+" CE Loss: "+str(cross_entropy_loss)+" Accuracy: "+str(acc))

    return weights

def main_function():
    # Load training data.
    # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    # 0.5 (so that the range is [-0.5,+0.5]).

    Ws, bs = initWeightsAndBiases()
    trainX = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28*28)).T
    trainY = np.load("fashion_mnist_train_labels.npy").reshape(-1,1).T
    testX = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28*28)).T
    testY = np.load("fashion_mnist_test_labels.npy").reshape(-1,1).T

    trainX = trainX / 255  
    testX = testX / 255

    trainX = trainX - 0.5
    testX = testX - 0.5

    # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    # On just the first 5 training examlpes, do numeric gradient check.
    # Use just the first return value ([0]) of fCE, which is the cross-entropy.
    # The lambda expression is used so that, from the perspective of
    # check_grad and approx_fprime, the only parameter to fCE is the weights
    # themselves (not the training data).
    # print(scipy.optimize.check_grad(lambda weights_: fCE(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_)[0], \
    #                                 lambda weights_: gradCE(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_), \
    #                                 weights))
    # print(scipy.optimize.approx_fprime(weights, lambda weights_: fCE(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_)[0], 1e-6))

    weights = train(trainX, trainY, weights, testX, testY, lr)

    cross_entropy_loss, Z, H, y_hat = fCE(testX, testY, weights)
    acc = accuracy(y_hat, testY)
    print(" Final Testing Loss: "+str(cross_entropy_loss)+" Final Testing Accuracy: "+str(acc))
    show_W0(weights)

main_function()