import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

inscale = np.zeros(16)
outscale = np.zeros(60)
inscale = torch.tensor(inscale, dtype=torch.float32)    

class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as file:
            for line in file:
                self.data.append([float(x) for x in line.strip().split(" ")])
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        self.inputs = self.data[:, 60:]
        self.outputs = self.data[:, :60]
        
        self.inputs = self.inputs / inscale
        # self.outputs = self.outputs / outscale
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 60)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss =  0
    pval = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pval.append(pred.cpu().numpy())
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    pval = np.concatenate(pval, axis = 0)

    print(f"Test Error: \n Avg loss: {test_loss} \n")
    # return (pval + 1) / 2 * 1200 - 600
    print(pval)
    return pval * np.array(outscale)

def gety(dataloader):
    yval = []
    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(device)
            y = y.cpu().numpy()
            yval.append(y)
    yval = np.concatenate(yval, axis = 0)
    # return (yval + 1) / 2 * 1200 - 600
    return yval



if __name__ == "__main__":
    
    test_data = TextDataset("fullmodeldata/subfixed.txt")
    test_dataloader = DataLoader(test_data, batch_size=100)
    
    a = gety(test_dataloader)
    full = np.loadtxt("fullmodeldata/subfixed.txt")
    
    # maxi = []
    # for i in range(1, len(full)):
    #     curr = abs(full[i] - full[i-1])
    #     # maxi = max(maxi, max(curr))
    #     for j in range(60): maxi.append(curr[j])

    
    # # filter data
    # ind = []
    # i = 1
    # while (i < len(full)):
    #     if max(abs(full[i] - full[i-1])) > 350: i += 50
    #     else: ind.append(i)
    #     i += 1
    #     if (i >= len(full)): break
    # data = []
    # for i in ind: data.append(full[i])
    # print(len(data))
    # np.savetxt("fullmodeldata/subfixed.txt", np.array(data))
    
    
    # #subtract and log
    # # a = a - (a[0] + a[1] + a[2] + a[3] + a[4]) / 5
    # print(np.amin(a))
    # a = a - np.amin(a) + 1000
    # a = np.log(a)
    # a = a - 7
    # data = []
    # for i in range(len(a)):
    #     curr = []
    #     for j in range(60): curr.append(a[i][j])
    #     for j in range(16): curr.append(full[i][j + 60])
    #     data.append(curr)
    # np.savetxt("fullmodeldata/logdata2.txt", np.array(data))
    # ##-2473.9802 min for fixed (600), -1308.9 for 350
    
    a = a.T
    # data = np.array(data).T
    
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7))
    n = 30
    for i in range(3):
        for j in range(5):
            axes[i][j].plot(a[n], label = "actual", linestyle="-")
            n += 1
            plt.legend()

    plt.tight_layout()
    plt.savefig("img")

