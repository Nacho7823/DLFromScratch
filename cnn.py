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
        dL_dF = loss * self.dfun(self.preactivation)
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
            
            if True:
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
            else:
                
                # fast convolution for dL_dX
                filter_rotated = np.flip(self.filters[n], axis=(1,2))
                for c in range(self.input_shape[0]):
                    dL_dX[c] += self.convolve2d(dL_dF[n], filter_rotated[c])
                
                    
                            
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

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    # La derivada es 1 - tanh(x)^2
    # Si ya tienes la salida (out = tanh(x)), puedes usar: 1 - out**2
    return 1 - np.tanh(x)**2

def stanh(x):
    return 2 * np.tanh(x / 2)

def dstanh(x):    
    return 2 * (1 - np.tanh(x / 2)**2)


def lecun_tanh(x):
    return 1.7159 * np.tanh((2/3) * x)

def dlecun_tanh(x):
    # Derivada usando la preactivación x
    return 1.14393 * (1 - np.tanh((2/3) * x)**2)


# error functions
def mse(y_true : np.ndarray, y_pred : np.ndarray):
    return np.mean((y_true - y_pred)**2)

def dmse(y_true : np.ndarray, y_pred : np.ndarray):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true : np.ndarray, y_pred : np.ndarray):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def dcross_entropy(y_true : np.ndarray, y_pred : np.ndarray):
    return -(y_true / (y_pred + 1e-9))

# f2 = CNNLayer(input_shape=(10, 20, 20), num_filters=15, filter_size=5, stride=1, padding=0, fun=stanh, dfun=dstanh)


learningRate = 0.01
f0 = CNNLayer(input_shape=(1, 28, 28), num_filters=5, filter_size=5, stride=1, padding=0, fun=stanh, dfun=dstanh)
#28 - 5 + 1 = 24
f1 = CNNLayer(input_shape=(5, 24, 24), num_filters=8, filter_size=5, stride=1, padding=0, fun=stanh, dfun=dstanh)
f3 = CNNLayer(input_shape=(8, 20, 20), num_filters=15, filter_size=3, stride=1, padding=0, fun=stanh, dfun=dstanh)
f4 = FFNLayer(input=15*18*18, hidden=10, fun=lecun_tanh, dfun=dlecun_tanh)
f5 = FFNLayer(input=10, hidden=10, fun=lecun_tanh, dfun=dlecun_tanh)

def train(inp, Y):
    global f0, f1, f2, f3, f4, f5, learningRate

    h0 = f0.forward(inp)
    h1 = f1.forward(h0)
    # h2 = f2.forward(h1)
    # h3 = f3.forward(h2)
    h3 = f3.forward(h1)
    h3_flat = h3.reshape((h3.shape[0]*h3.shape[1]*h3.shape[2], 1))
    h4 = f4.forward(h3_flat)
    out = f5.forward(h4)

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
    
    # if loss < 0.09:
    #     learningRate = 0.4
    
    
    # print(f"dloss: {dloss.T}")

    dloss = dloss.reshape((dloss.shape[0], 1))
    lam5 = f5.backward(dloss, learningRate)
    
    lam4 = f4.backward(lam5, learningRate)
    
    lam4c = lam4.reshape((f3.outputShape[0], f3.outputShape[1], f3.outputShape[2]))
    
    lam3 = f3.backward(lam4c, learningRate)
    
    # lam2 = f2.backward(lam3, learningRate)
    # lam1 = f1.backward(lam2, learningRate)
    
    lam1 = f1.backward(lam3, learningRate)
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
    
    for i in range(len(trainData[:1000])):
        
        inp = trainData[i] / 255.0
        inp = inp.reshape((1, 28, 28))
        
        
        Y = np.zeros((10, 1))
        Y[trainLabels[i]] = 1
        
        loss = train(inp, Y)
        
        print(f"Epoch {epoch+1}, Sample {i+1} trained. Loss: {loss}")
        # print(f"Epoch {epoch+1}, Sample {i+1} trained. Loss: {loss}", end='\r')
        
    print(f"Epoch {epoch+1} completed.")
    


# test on first 10 images
for i in range(10):
    inp = testData[i].reshape((1, 28, 28)) / 255.0
    
    h5 = f0.forward(inp)
    h4 = f1.forward(h5)
    # h3 = f2.forward(h4)
    # h2 = f3.forward(h3)
    h2 = f3.forward(h4)
    h2_flat = h2.reshape((h2.shape[0]*h2.shape[1]*h2.shape[2], 1))
    h1 = f4.forward(h2_flat)
    out = f5.forward(h1)
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