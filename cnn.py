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
  
  
  
  
class CNNLayer:
    
    def __init__(self, input_shape, num_filters, filter_size, stride, padding, fun, dfun):
        self.input_shape = input_shape  # (channels, height, width)
        self.num_filters = num_filters  # characteristic maps
        self.filter_size = filter_size  # assuming square filters
        self.fun = fun
        self.dfun = dfun
        
        self.preactivation = np.zeros((num_filters, input_shape[1] - filter_size + 1, input_shape[2] - filter_size + 1))
        self.filters = np.random.randn(num_filters, input_shape[0], filter_size, filter_size) * 0.1
        self.biases = np.zeros((num_filters, 1))
        
    def forward(self, input):
        self.input = input


        # Implement forward pass for convolutional layer
        self.output = np.zeros((self.num_filters, self.input_shape[1] - self.filter_size + 1, self.input_shape[2] - self.filter_size + 1))
        self.outputShape = self.output.shape
        
        # cross-correlation operation
        for n in range(self.num_filters):
            
            for i in range(0, self.outputShape[1]):
                for j in range(0, self.outputShape[2]):
                    
                    region = self.input[:, i:i+self.filter_size, j:j+self.filter_size]
                    
                    self.preactivation[n, i, j] = np.sum(region * self.filters[n]) + self.biases[n]
        
        self.output = self.fun(self.preactivation)
        
        return self.output

    def backward(self, loss, learningRate):
        dL_dF = loss * self.dfun(self.output)
        # Initialize gradients
        dL_dW = np.zeros_like(self.filters)
        dL_dB = np.zeros_like(self.biases)
        dL_dX = np.zeros_like(self.input)
        
        # Compute gradients
        for n in range(self.num_filters):
            for i in range(0, self.filters.shape[2]):
                for j in range(0, self.filters.shape[3]):
                    
                    region = self.input[:, i:i + dL_dF.shape[1], j:j + dL_dF.shape[2]]
                    # dL_dW 1, 8, 24, 24
                    # region 8, 3, 3
                    
                    s = region * dL_dF[n]
                    s1 = np.sum(s, axis=(1,2))
                    dL_dW[n, :, i, j] += s1
                    
            dL_dB[n] = np.sum(dL_dF[n])
            
        for n in range(self.num_filters):
            for i in range(0, self.filter_size):
                for j in range(0, self.filter_size):
                    
                    if False:
                        for h in range(dL_dF.shape[1]):
                            for w in range(dL_dF.shape[2]):
                                dL_dX[:, i + h, j + w] += self.filters[n, :, i, j] * dL_dF[n, h, w]
                    else:
                        region = dL_dF[n, :, :]
                        s = self.filters[n, :, i, j]
                        s = s.reshape((s.shape[0],1,1))
                        s = s * region
                        
                        
                        dL_dX[:, i:i + region.shape[0], j:j + region.shape[1]] += s
                    
                            
        # Update weights and biases
        self.filters -= learningRate * dL_dW
        self.biases -= learningRate * dL_dB
                            
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

learningRate = 0.001
f0 = CNNLayer(input_shape=(1, 28, 28), num_filters=8, filter_size=3, stride=1, padding=0, fun=relu, dfun=drelu)
f1 = CNNLayer(input_shape=(8, 26, 26), num_filters=1, filter_size=3, stride=1, padding=0, fun=relu, dfun=drelu)
f2 = FFNLayer(input=1*24*24, hidden=10, fun=relu, dfun=drelu)

def train(inp, Y):
    global f0, f1

    h0 = f0.forward(inp)
    h1 = f1.forward(h0)
    h1_flat = h1.reshape((h1.shape[0]*h1.shape[1]*h1.shape[2], 1))
    out = f2.forward(h1_flat)

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
    
    lam2c = lam2.reshape((f1.outputShape[0], f1.outputShape[1], f1.outputShape[2]))
    
    lam1 = f1.backward(lam2c, learningRate)
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
        inp = inp.reshape((1, 28, 28))
        
        
        Y = np.zeros((10, 1))
        Y[trainLabels[i]] = 1.0
        
        loss = train(inp, Y)
        
        print(f"Epoch {epoch+1}, Sample {i+1} trained. Loss: {loss}")
        # print(f"Epoch {epoch+1}, Sample {i+1} trained. Loss: {loss}", end='\r')
        
    print(f"Epoch {epoch+1} completed.")
    


# test on first 10 images
for i in range(10):
    inp = testData[i].reshape((1, 28, 28)) / 255.0
    out = f2.forward(f1.forward(f0.forward(inp)))
    pred = np.argmax(out)
    # print(f"Image {i}, True Label: {testLabels[i]}, Predicted: {pred}, Output: {out.T}")
    print(f"Image {i}, True Label: {testLabels[i]}, Predicted: {pred}")
    
    
# save weights

# np.save('models/f0_weights.npy', f0.weights)
# np.save('models/f0_bias.npy', f0.bias)
# np.save('models/f1_weights.npy', f1.weights)
# np.save('models/f1_bias.npy', f1.bias)
# np.save('models/f2_weights.npy', f2.weights)
# np.save('models/f2_bias.npy', f2.bias)