import numpy as np

def f(x):
    return np.maximum(0, x)

def df(x):
    return (x > 0).astype(int)

def loss(ytrue, ypred):
    return ((ytrue - ypred) ** 2).mean()

class Neuralnetw:
    def __init__(self):
        self.win = np.random.randn(16, 128) * 0.1
        self.bin = np.random.randn(1, 128) * 0.1
        self.w = np.random.randn(128, 128) * 0.1
        self.b = np.random.randn(1, 128) * 0.1
        self.wout = np.random.randn(128, 60) * 0.1
        self.bout = np.random.randn(1, 60) * 0.1
    
    def feedforward(self, x):
        #weights column j = neuron j
        self.a1 = np.dot(x, self.win) + self.bin
        self.ans1 = f(self.a1) #ans1[1][128] neuron j output
        self.ans2 = f(np.dot(self.ans1, self.w) + self.b) #ans2[1][128]
        self.ans = np.dot(self.ans2, self.wout) + self.bout #ans[1][60]
        return self.ans
    
    def train(self, x, y, rate):
        for i in range(100000): 
            tot = 0  
            for xi, yi in zip(x, y): 
                xi = xi.reshape(1, -1)  # xi[1][16]
                yi = yi.reshape(1, -1)  # yi[1][60]

                output = self.feedforward(xi)
                l = loss(output, yi)
                tot += l

                dL = 2 * (output - yi) / 60 #dL[1][60] d loss d output from neuron j

                # output layer ans[i] = sum j(wout[j][i] * ans2[j]) + b[i], ans2[j] * dL[i]
                dwout = np.dot(self.ans2.T, dL) #dwout[128][60] d loss d weight i neuron j
                dbout = np.sum(dL, axis=0) 

                # second layer df(...) * wout[j][i]
                dans2 = np.dot(dL, self.wout.T) * df(self.ans2) #dans2[1][128]
                dw = np.dot(self.ans1.T, dans2)
                db = np.sum(dans2, axis=0)

                # input layer
                dans1 = np.dot(dans2, self.w.T) * df(self.ans1)
                dwin = np.dot(xi.T, dans1)
                dbin = np.sum(dans1, axis=0)

                # update
                self.wout -= rate * dwout
                self.bout -= rate * dbout
                self.w -= rate * dw
                self.b -= rate * db
                self.win -= rate * dwin
                self.bin -= rate * dbin
            if (i % 100 == 0):
                print("Loss : ", tot)
            


# data = np.loadtxt("/Users/sissi/Downloads/2/anyskin/data/data_2025-01-17_16-48-54.txt", delimiter=" ")
# datain = np.zeros((data.shape[0], 16))
# dataout = np.zeros((data.shape[0], 60))
# for i in range(data.shape[0]):
#     for j in range(76):
#         if (j < 16):
#             datain[i][j] = data[i][j]
#         else:
#             dataout[i][j-16] = data[i][j]
# datain = (datain - np.min(datain)) / (np.max(datain) - np.min(datain))

# random data
datain = np.random.rand(100, 16)
dataout = np.random.rand(100, 60)

# Train neural network
network = Neuralnetw()
network.train(datain, dataout, 0.000001)

