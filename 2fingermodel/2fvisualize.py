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

inscale = torch.tensor([1922, 2902, 2597, 3897, 1891, 3998, 2104, 3099])
outscale = torch.tensor([64.8, 65.7, 185.37, 60, 71, 173, 67.5, 67.8, 182.46, 56.1, 72.6, 163.1, 63, 82, 208, 106, 67.8, 108, 97, 54, 97, 95, 113, 115, 95, 86, 95, 62, 123, 138])
        

class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as file:
            for line in file:
                self.data.append([float(x) for x in line.strip().split(" ")])
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        self.inputs = self.data[:, 30:]
        self.outputs = self.data[:, :30]
        self.outputs = self.outputs - self.outputs[0]
        
        # self.inputs = self.inputs / inscale
        # self.outputs = self.outputs / outscale
        
        self.inputs = 2 * (self.inputs - 1800) / (3100 - 1800) - 1 
        # self.outputs = 2 * (self.outputs + 100) / 350 - 1 #scale iwth -100 to 250, 1800 - 3100
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 30)
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
    return (pval + 1) / 2 * 350 - 100
    # return pval * np.array(outscale)

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
    test_data = TextDataset("2fingermodel/2fmodeldata/0skin.txt")#503
    test_dataloader = DataLoader(test_data, batch_size=100)
    
    t1 = TextDataset("2fingermodel/2fmodeldata/exp1skin.txt")#503
    td1 = DataLoader(t1, batch_size=100)
    
    s1 = TextDataset("2fingermodel/2fmodeldata/1skinnocontact.txt")#503
    ts1 = DataLoader(s1, batch_size=100)

    model.load_state_dict(torch.load("2fingermodel/2fmodels/2fmodelmore.pth", map_location = 'cpu'))
    b = test(test_dataloader, model, loss_fn)
    a = gety(test_dataloader)
    x = gety(ts1)
    
    tr1 = gety(td1).T
    
    # diff = a - b
    # ans = x + diff
    # ans = ans.T
    # a = a
    print(b[0])
    a = a.T
    b = b - b[0]
    b = b.T
    
    

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7))
    n = 15
    for i in range(3):
        for j in range(5):
            axes[i][j].plot(b[n], label = "1 skin", linestyle="--")
            axes[i][j].plot(a[n], label = "2 skins", linestyle="-")
            # axes[i][j].plot(ans[n], label = "predicted", linestyle="-.")
            n += 1
            plt.legend()

    fig.canvas.draw()
    for i in range(3):
        for j in range(5):
            axes[i][j].set_ylim(min(axes[i][j].get_ylim()[0], -30), max(axes[i][j].get_ylim()[1], 30))

    plt.tight_layout()
    plt.savefig("img")
