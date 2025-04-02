import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import statistics


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
outscale = torch.tensor(outscale, dtype=torch.float32)

class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as file:
            for line in file:
                self.data.append([float(x) for x in line.strip().split(" ")])
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        self.inputs = self.data[:, 60:]
        self.outputs = self.data[:, :60]
        baseline = (self.outputs[0] + self.outputs[1] + self.outputs[2] + self.outputs[3] + self.outputs[4]) / 5
        self.outputs = self.outputs - baseline
        
        if inscale[0] == 0:
            for i in range(16):
                inscale[i] = (max(abs(self.inputs.T[i].max()), abs(self.inputs.T[i].min())))
            for i in range(60):
                outscale[i] = (max(abs(self.outputs.T[i].max()), abs(self.outputs.T[i].min())))
                   
        self.inputs = self.inputs / inscale
        self.outputs = self.outputs / outscale
        
        # self.inputs = 2 * (self.inputs - 1800) / (4000 - 1800) - 1 
        # self.outputs = 2 * (self.outputs + 600) / 1200 - 1 #scale iwth -600 to 600, 1800 - 4000
        

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.8)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch + 1) * len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
    # return (pval + 1) * 1200 / 2 - 600
    return pval * outscale.cpu().numpy()

def gety(dataloader):
    yval = []
    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(device)
            y = y.cpu().numpy()
            yval.append(y)
    yval = np.concatenate(yval, axis = 0)
    return yval * outscale.cpu().numpy()



if __name__ == "__main__":
    training_data = TextDataset("fullmodeldata/moredatafull.txt")
    test_data = TextDataset("fullmodeldata/moredatafull.txt")

    train_dataloader = DataLoader(training_data, batch_size=100)
    test_dataloader = DataLoader(test_data, batch_size=100)

    scales = np.zeros(76)
    for i in range(16):
        scales[i] = inscale[i]
    for i in range(60):
        scales[16+ i] = outscale[i]
    np.savetxt("scale.txt", scales)
    
    print(gety(test_dataloader))

    epochs = 10000
    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        if (t % 50 == 0): test(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), "fullmodelscaled1.pth") 

    res = test(test_dataloader, model, loss_fn).T
    act = gety(test_dataloader).T
    plt.plot(res[3], label = "predicted", linestyle="-.")
    plt.plot(act[3], label = "actual", linestyle="-")
    plt.legend()
    plt.show()
