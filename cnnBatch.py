import numpy as np

class FFNLayer:

    def __init__(self, input ,hidden, fun, dfun, batchsize):
        self.input = input
        self.hidden = hidden
        self.batchsize = batchsize
        # self.weights = np.ones((hidden, input))
        self.weights = (np.random.rand(hidden, input) - 0.5) * 2/np.sqrt(input)
        
        self.bias = np.zeros((hidden, 1))
        self.values = np.zeros((hidden, batchsize))
        self.preactivation = np.zeros((hidden, batchsize))
        self.fun = fun
        self.dfun = dfun

    def forward(self, I):
        # I shape: (in_dim, batch)
        self.input = I
        self.preactivation = self.weights @ self.input + self.bias
        self.values = self.fun(self.preactivation)
        return self.values

    def backward(self, loss, learningRate):
        # loss shape: (hidden, batch)
        dL_dF = loss * self.dfun(self.preactivation)
        dL_dW = dL_dF @ self.input.T
        dL_dB = dL_dF.sum(axis=1, keepdims=True)
        dL_dX = self.weights.T @ dL_dF
        
        # Simple Adam-like moment buffers initialized on first use
        if not hasattr(self, 'weights_m'):
            self.weights_m = np.zeros_like(self.weights)
            self.weights_v = np.zeros_like(self.weights)
            self.bias_m = np.zeros_like(self.bias)
            self.bias_v = np.zeros_like(self.bias)
            self.t = 0
        
        self.t += 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        self.weights_m = beta1 * self.weights_m + (1 - beta1) * dL_dW
        self.weights_v = beta2 * self.weights_v + (1 - beta2) * (dL_dW ** 2)
        m_hat = self.weights_m / (1 - beta1**self.t)
        v_hat = self.weights_v / (1 - beta2**self.t)

        self.bias_m = beta1 * self.bias_m + (1 - beta1) * dL_dB
        self.bias_v = beta2 * self.bias_v + (1 - beta2) * (dL_dB ** 2)
        bm_hat = self.bias_m / (1 - beta1**self.t)
        bv_hat = self.bias_v / (1 - beta2**self.t)

        self.weights -= learningRate * m_hat / (np.sqrt(v_hat) + eps)
        self.bias -= learningRate * bm_hat / (np.sqrt(bv_hat) + eps)
        
        return dL_dX
  
  
  
  
class CNNLayer:
    
    def __init__(self, input_shape, num_filters, filter_size, stride, padding, fun, dfun):
        # input_shape: (channels, height, width)
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.fun = fun
        self.dfun = dfun
        
        # filters: (num_filters, channels, f, f)
        self.filters = np.random.randn(num_filters, input_shape[0], filter_size, filter_size) * 0.1
        self.biases = np.zeros((num_filters, 1))
        # placeholders (will be initialized in forward based on batch)
        self.preactivation = None
        self.output = None
        self.outputShape = None
        
    def forward(self, input):
        # input shape: (C, H, W, B)
        self.input = input
        C, H, W, B = input.shape
        outH = H - self.filter_size + 1
        outW = W - self.filter_size + 1
        self.preactivation = np.zeros((self.num_filters, outH, outW, B))
        self.output = np.zeros((self.num_filters, outH, outW, B))
        self.outputShape = self.output.shape
        
        # cross-correlation over batch
        for n in range(self.num_filters):
            f = self.filters[n]  # shape (C, f, f)
            # compute for each spatial location
            for i in range(outH):
                for j in range(outW):
                    region = input[:, i:i+self.filter_size, j:j+self.filter_size, :]  # (C, f, f, B)
                    # elementwise multiply, sum over C,f,f -> result shape (B,)
                    s = np.sum(region * f[:, :, :, None], axis=(0,1,2))
                    # add bias
                    self.preactivation[n, i, j, :] = s + self.biases[n]
        
        self.output = self.fun(self.preactivation)
        return self.output

    def backward(self, loss, learningRate):
        # loss shape: (num_filters, outH, outW, B)
        dL_dF = loss * self.dfun(self.preactivation)
        C, H, W, B = self.input.shape
        outH = dL_dF.shape[1]
        outW = dL_dF.shape[2]
        
        # Initialize gradients
        dL_dW = np.zeros_like(self.filters)
        dL_dB = np.zeros_like(self.biases)
        dL_dX = np.zeros_like(self.input)
        
        # Gradients for filters and biases
        for n in range(self.num_filters):
            # bias gradient: sum over spatial and batch
            dL_dB[n, 0] = np.sum(dL_dF[n])
            for c in range(C):
                for i in range(self.filter_size):
                    for j in range(self.filter_size):
                        # patch over spatial positions and batch
                        patch = self.input[c, i:i+outH, j:j+outW, :]  # shape (outH, outW, B)
                        s = patch * dL_dF[n]  # broadcasts, shape (outH, outW, B)
                        dL_dW[n, c, i, j] = np.sum(s)
        
        # Gradient w.r.t. input
        for n in range(self.num_filters):
            for c in range(C):
                for i in range(self.filter_size):
                    for j in range(self.filter_size):
                        # add contribution of this filter channel to dL_dX
                        dL_dX[c, i:i+outH, j:j+outW, :] += self.filters[n, c, i, j] * dL_dF[n]
        
        # Average gradients by batch size to stabilize updates
        batch_factor = dL_dF.shape[3] if dL_dF.shape[3] > 0 else 1
        dL_dW /= batch_factor
        dL_dB /= batch_factor
        
        # Update weights and biases (SGD)
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
    return 1 - np.tanh(x)**2

def stanh(x):
    return 2 * np.tanh(x / 2)

def dstanh(x):    
    return 2 * (1 - np.tanh(x / 2)**2)

def lecun_tanh(x):
    return 1.7159 * np.tanh((2/3) * x)

def dlecun_tanh(x):
    return 1.14393 * (1 - np.tanh((2/3) * x)**2)

# error functions
def mse(y_true : np.ndarray, y_pred : np.ndarray):
    return np.mean((y_true - y_pred)**2)

def dmse(y_true : np.ndarray, y_pred : np.ndarray):
    # keep same scaling as original cnnBatch (divide by total size)
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true : np.ndarray, y_pred : np.ndarray):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def dcross_entropy(y_true : np.ndarray, y_pred : np.ndarray):
    return -(y_true / (y_pred + 1e-9))

# network setup
bsize = 32
learningRate = 0.001
f0 = CNNLayer(input_shape=(1, 28, 28), num_filters=5, filter_size=5, stride=1, padding=0, fun=stanh, dfun=dstanh)
# 28 - 5 + 1 = 24
f1 = CNNLayer(input_shape=(5, 24, 24), num_filters=8, filter_size=5, stride=1, padding=0, fun=stanh, dfun=dstanh)
f3 = CNNLayer(input_shape=(8, 20, 20), num_filters=15, filter_size=3, stride=1, padding=0, fun=stanh, dfun=dstanh)
f4 = FFNLayer(input=15*18*18, hidden=10, fun=lecun_tanh, dfun=dlecun_tanh, batchsize=bsize)
f5 = FFNLayer(input=10, hidden=10, fun=lecun_tanh, dfun=dlecun_tanh, batchsize=bsize)

def train(inp, Y):
    global f0, f1, f2, f3, f4, f5, learningRate

    # inp: (C, H, W, B)
    h0 = f0.forward(inp)                     # (5, 24,24,B)
    h1 = f1.forward(h0)                      # (8, 20,20,B)
    h3 = f3.forward(h1)                      # (15, 18,18,B)
    # flatten to (in_dim, batch)
    h3_flat = h3.reshape((h3.shape[0]*h3.shape[1]*h3.shape[2], h3.shape[3]))
    h4 = f4.forward(h3_flat)                 # (10, B)
    out = f5.forward(h4)                     # (10, B)

    # Y expected shape: (10, B)
    loss = mse(Y, out)
    dloss = dmse(Y, out)
    
    lam5 = f5.backward(dloss, learningRate)
    lam4 = f4.backward(lam5, learningRate)
    
    lam4c = lam4.reshape((f3.filters.shape[0], f3.outputShape[1], f3.outputShape[2], f3.outputShape[3]))
    
    lam3 = f3.backward(lam4c, learningRate)
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

#train for 1 epoch (example)
for epoch in range(1):
    
    for i in range(len(trainData[:20000]) // bsize):
        
        batch = trainData[i*bsize:(i+1)*bsize] / 255.0  # (B,28,28)
        # convert to shape (C, H, W, B) where C=1
        inp = np.transpose(batch, (1,2,0))[None, ...]  # (1,28,28,B)
        
        Y = np.zeros((10, bsize))
        labels = trainLabels[i*bsize:(i+1)*bsize]
        Y[labels, np.arange(bsize)] = 1.0
        
        loss = train(inp, Y)
        
        print(f"Epoch {epoch+1}, Batch {i+1} trained. Loss: {loss}")
        
    print(f"Epoch {epoch+1} completed.")
    

# test on first 10 images
for i in range(10):
    sample = testData[i] / 255.0  # (28,28)
    inp = sample[None, :, :, None]  # (1,28,28,1)
    
    h5 = f0.forward(inp)
    h4 = f1.forward(h5)
    h2 = f3.forward(h4)
    h2_flat = h2.reshape((h2.shape[0]*h2.shape[1]*h2.shape[2], h2.shape[3]))
    h1 = f4.forward(h2_flat)
    out = f5.forward(h1)
    pred = np.argmax(out[:, 0])
    print(f"Image {i}, True Label: {testLabels[i]}, Predicted: {pred}")
    
# save weights (optional)
# np.save('models/f0_filters.npy', f0.filters)
# np.save('models/f0_bias.npy', f0.biases)
