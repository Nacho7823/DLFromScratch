'''
X = [x0 x1 x2]

W = [w00 w01 w02]
    [w10 w11 w12]
    [w20 w21 w22]
    [w30 w31 w32]

x0*w00 + x1*w01 + x2*w02 = f00
x0*w10 + x1*w11 + x2*w12 = f01
x0*w20 + x1*w21 + x2*w22 = f02
x0*w30 + x1*w31 + x2*w32 = f03

F0 = W * X
H0 = o(F0)

Q = [q00 q01 q02 q03]
    [q10 q11 q12 q13]

h00 * q00 + h01 * q01 + h02 * q02 + h03 * q03 = f10
h00 * q10 + h01 * q11 + h02 * q12 + h03 * q13 = f11

F1 = Q * H0
H1 = o(F1)

L = H1 - y**2

dL/dH = 2(H - y)
dH/dF = o(F)'

dF/qij = h0j

dF/dh0j = qij

dh0j/df0j = o(f0j)'

df0j/wij = xj

df0j/xj


'''
import numpy as np

class FFNLayer:


    def __init__(self, input ,hidden, fun, dfun):
        self.input = input
        self.hidden = hidden
        # self.weights = np.ones((hidden, input))
        self.weights = (np.random.rand(hidden, input) - 0.5) * 2/np.sqrt(input)
        
        self.bias = np.zeros((hidden, 1))
        self.values = np.zeros((1,hidden))
        self.preactivation = np.zeros((1,hidden))
        self.fun = fun
        self.dfun = dfun

    def forward(self, I):
        self.input = I.reshape((I.shape[0],1))
        self.preactivation = self.weights @ self.input + self.bias
        self.values = self.fun(self.preactivation)
        return self.values

    def backward(self, loss, learningRate):
        dL_dF = loss * self.dfun(self.preactivation)
        dL_dW = dL_dF @ self.input.T
        dL_dB = dL_dF
        dL_dX = self.weights.T @ dL_dF
        
        # print(f"dL_dW: {dL_dW}")
        # print(f"dL_dB: {dL_dB}")
        
        
        self.weights -= learningRate * dL_dW
        self.bias -= learningRate * dL_dB
        
        
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

def softmax(x : np.ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def dsoftmax(x : np.ndarray):
    s = softmax(x)
    return s * (1 - s)

# error functions
def mse(y_true : np.ndarray, y_pred : np.ndarray):
    return np.mean((y_true - y_pred)**2)

def dmse(y_true : np.ndarray, y_pred : np.ndarray):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true : np.ndarray, y_pred : np.ndarray):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def dcross_entropy(y_true : np.ndarray, y_pred : np.ndarray):
    return -(y_true / (y_pred + 1e-9))

hid = 50
learningRate = 0.001
f0 = FFNLayer(28*28, hid, relu, drelu)
f1 = FFNLayer(hid, hid, relu, drelu)
f2 = FFNLayer(hid, 10, sigmoid, dsigmoid)

def train(inp, Y):
    global f0, f1, f2

    h0 = f0.forward(inp)
    h1 = f1.forward(h0)
    out = f2.forward(h1)

    Y = Y.reshape((out.shape[0], 1))
    
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

    dloss = dloss.reshape((dloss.shape[0], 1))
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
for epoch in range(1):
    
    for i in range(len(trainData)):
        
        inp = trainData[i] / 255.0
        inp = inp.reshape((1, 28*28)).T
        
        
        Y = np.zeros((10, 1))
        Y[trainLabels[i]] = 1.0
        
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