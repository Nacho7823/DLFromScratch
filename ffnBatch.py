#TODO

import numpy as np

class FFNLayer:


    def __init__(self, input ,hidden, fun, dfun, batchsize):
        self.input = input
        self.hidden = hidden
        self.batchsize = batchsize
        # self.weights = np.ones((hidden, input))
        self.weights = (np.random.rand(hidden, input) - 0.5) * 2/np.sqrt(input)
        
        self.bias = np.zeros((hidden, 1))
        self.values = np.zeros((batchsize, hidden))
        self.preactivation = np.zeros((batchsize, hidden))
        self.fun = fun
        self.dfun = dfun

    def forward(self, I):
        self.input = I
        # print(f"Input shape: {self.input.shape}")
        # print(f"Weights shape: {self.weights.shape}")
        self.preactivation = self.weights @ self.input + self.bias
        self.values = self.fun(self.preactivation)
        return self.values

    def backward(self, loss, learningRate):
        dL_dF = loss * self.dfun(self.preactivation)
        dL_dW = dL_dF @ self.input.T
        dL_dB = dL_dF.sum(axis=1, keepdims=True)
        dL_dX = self.weights.T @ dL_dF
        
        # print(f"dL_dW: {dL_dW}")
        # print(f"dL_dB: {dL_dB}")
        
        
        # optimice using adam
        self.weights_m = np.zeros_like(self.weights)
        self.weights_v = np.zeros_like(self.weights)
        self.bias_m = np.zeros_like(self.bias)
        self.bias_v = np.zeros_like(self.bias)
        
        momentumCoeff = 0.5
        rmsCoeff = 0.6
        epsilon = 1e-8
        
        self.weightsM = momentumCoeff * self.weights_m + (1 - momentumCoeff) * dL_dW
        self.weightsV = rmsCoeff * self.weights_v + (1 - rmsCoeff) * (dL_dW ** 2)
        
        self.biasM = momentumCoeff * self.bias_m + (1 - momentumCoeff) * dL_dB
        self.biasV = rmsCoeff * self.bias_v + (1 - rmsCoeff) * (dL_dB ** 2)
        
    
        self.weights -= learningRate * self.weightsM / (np.sqrt(self.weightsV) + epsilon)
        self.bias -= learningRate * self.biasM / (np.sqrt(self.biasV) + epsilon)
        
        return dL_dX
  
# activation functions  
def relu(x : np.ndarray):
    return np.maximum(0, x)

def drelu(x : np.ndarray):
    return np.where(x > 0, 1.00, 0.00)

def sigmoid(x : np.ndarray):
    return 1 / (1 + np.exp(-x))
def dsigmoid(x : np.ndarray):
    s = sigmoid(x)
    return s * (1 - s)

# def softmax(x : np.ndarray):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)
# def dsoftmax(x : np.ndarray):
#     s = softmax(x)
#     return s * (1 - s)

# error functions
def mse(y_true : np.ndarray, y_pred : np.ndarray):
    return np.mean((y_true - y_pred)**2)

def dmse(y_true : np.ndarray, y_pred : np.ndarray):
    return 2 * (y_pred - y_true)

# def cross_entropy(y_true : np.ndarray, y_pred : np.ndarray):
#     return -np.sum(y_true * np.log(y_pred + 1e-9))

# def dcross_entropy(y_true : np.ndarray, y_pred : np.ndarray):
#     return -(y_true / (y_pred + 1e-9))




#------------------------------------------------------------------------------------------------
# Example usage on MNIST dataset
#------------------------------------------------------------------------------------------------
hid = 50
bsize = 64
learningRate = 0.001
f0 = FFNLayer(28*28, hid, relu, drelu, bsize)
f1 = FFNLayer(hid, hid, relu, drelu, bsize)
f2 = FFNLayer(hid, 10, sigmoid, dsigmoid, bsize)

def train(inp, Y):
    global f0, f1, f2

    h0 = f0.forward(inp)
    h1 = f1.forward(h0)
    out = f2.forward(h1)

    # Y = Y.reshape((out.shape[0], 1))
    
    # print(f"Output: {out.T}, Target: {Y.T}")
    # Mean Squared Error Loss
    # loss = np.sum((out - Y)**2)
    # dloss = 2*(out - Y)
    
    # Cross Entropy Loss
    # loss = cross_entropy(Y, out)
    # dloss = dcross_entropy(Y, out)
    loss = mse(Y, out)
    dloss = dmse(Y, out)
    
    
    # print(f"dloss: {dloss.T}")

    # dloss = dloss.reshape((dloss.shape[0], 1))
    lam2 = f2.backward(dloss, learningRate)
    lam1 = f1.backward(lam2, learningRate)
    f0.backward(lam1, learningRate)
    return loss
    

# open data mnist digits (train-image.idx3-ubyte, train-labels.idx1-ubyte)
import idx2numpy
import matplotlib.pyplot as plt

trainData = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
trainLabels = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')
testData = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
testLabels = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')

print(f"Data shape: {trainData.shape}, Labels shape: {trainLabels.shape}")

# print one with matplotlib

# plt.imshow(trainData[0], cmap='gray')
# plt.title(f"Label: {trainLabels[0]}")
# plt.show()


#train for 5 epochs
for epoch in range(10):
    
    for i in range(len(trainData) // bsize):
        
        # batch of 4
        inp = trainData[i*bsize:(i+1)*bsize] / 255.0
        inp = inp.reshape((bsize, 28*28)).T
        
        Y = np.zeros((10, bsize))
        labels = trainLabels[i*bsize:(i+1)*bsize]
        
        Y[labels, np.arange(bsize)] = 1.0
        
        loss = train(inp, Y)
        
        print(f"Epoch {epoch+1}, Sample {i+1} trained. Loss: {loss}")
        # print(f"Epoch {epoch+1}, Sample {i+1} trained. Loss: {loss}", end='\r')
        
    print(f"Epoch {epoch+1} completed.")
    


# test on first 10 images
for i in range(10):
    inp = testData[i].reshape((28*28, 1)) / 255.0
    out = f2.forward(f1.forward(f0.forward(inp)))
    pred = np.argmax(out)
    # print(f"Image {i}, True Label: {testLabels[i]}, Predicted: {pred}, Output: {out.T}")
    print(f"Image {i}, True Label: {testLabels[i]}, Predicted: {pred}")
    
    
# save weights

np.save('models/f0_weights.npy', f0.weights)
np.save('models/f0_bias.npy', f0.bias)
np.save('models/f1_weights.npy', f1.weights)
np.save('models/f1_bias.npy', f1.bias)
np.save('models/f2_weights.npy', f2.weights)
np.save('models/f2_bias.npy', f2.bias)