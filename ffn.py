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
        self.weights = np.ones((hidden, input))
        print("weights: ", self.weights.shape)
        self.bias = np.ones((hidden, 1))
        print("bias: ", self.bias.shape)
        self.values = np.zeros((1,hidden))
        print("values: ", self.values.shape)
        self.fun = fun
        self.dfun = dfun

    def forward(self, I):
        print("I: ", I.shape)
        self.input = I.reshape((I.shape[0],1))
        print("input: ", self.input.shape)
        print("weights: ", self.weights.shape)
        self.values = self.fun(self.weights @ self.input + self.bias)
        print("values: ", self.values.shape)
        return self.values

    def backward(self, loss, learningRate):
        loss = self.dfun(loss)
        print(loss.shape)
        dL_dW = loss @ self.input.T
        dL_dB = loss
        dL_dX = self.weights.T @ loss

        print(f"dL_dW: {dL_dW}")
        print(f"dL_dB: {dL_dB}")
        print(self.input.T)
        print(f"dL_dX: {dL_dX}")
        self.weights -= learningRate * dL_dW
        self.bias -= learningRate * dL_dB
        
        return dL_dX
    
def relu(x):
    # if x < 0:
    #     x = x * 0
    return x

def drelu(x):
    # if x > 0:
    #     return x / x #1
    # else:
    #     return 0
    return x


def run(inp, Y):
    print("f0")
    f0 = FFNLayer(inp.shape[0], 4, relu, drelu)
    print("f1")
    f1 = FFNLayer(4, 2, relu, drelu)

    print("forward0")
    h0 = f0.forward(inp)
    print("forward1")
    out = f1.forward(h0)


    print(f"out: {out}")


    # loss = (sum(out - expected)) ** 2
    # print(f"loss: {loss}")
    Y = Y.reshape((out.shape[0], 1))

    dloss = 2*(out - Y)

    dloss = dloss.reshape((dloss.shape[0], 1))

    lam1 = f1.backward(dloss, 0.1)
    lam0 = f0.backward(lam1, 0.1)
    


i = np.array([2,1,2])
e = np.array([2,2])
run(i,e)