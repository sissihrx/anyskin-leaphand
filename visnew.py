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
    return pval

def gety(dataloader):
    yval = []
    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(device)
            y = y.cpu().numpy()
            yval.append(y)
    yval = np.concatenate(yval, axis = 0)
    return yval


if __name__ == "__main__":
    
    # model.load_state_dict(torch.load("pbmodel350.pth", map_location = 'cpu'))
    # scale = np.loadtxt("pbscale350.txt")
    model.load_state_dict(torch.load("pbmodelfixed.pth", map_location = 'cpu'))
    scale = np.loadtxt("pbscalefixed.txt")
    for i in range(16): inscale[i] = scale[i]
    
    test_data = TextDataset("fullmodeldata/temp.txt")
    test_dataloader = DataLoader(test_data, batch_size=100)
    b = test(test_dataloader, model, loss_fn)
    a = gety(test_dataloader)

    #a is actual, b is predicted, unnormalize b and subtract baseline from a 
    a = a - (a[0] + a[1] + a[2] + a[3] + a[4]) / 5
    # b = b + 7
    b = b + 8
    b = np.exp(b)
    b = b - 2473.9802 - 1000
    # b = b - 1308.9 - 1000
    b = b - b[0]
    a = a[2000:3000, :]
    b = b[2000:3000, :]
    a = a.T
    b = b.T
    
    #plot pred vs actual
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7))
    n = 15
    for i in range(3):
        for j in range(5):
            axes[i][j].plot(b[n], label = "predicted", linestyle="--")
            axes[i][j].plot(a[n], label = "actual", linestyle="-")
            n += 1
            plt.legend()

    plt.tight_layout()
    plt.savefig("img1")
