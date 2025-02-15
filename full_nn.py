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


class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as file:
            for line in file:
                self.data.append([float(x) for x in line.strip().split(" ")])
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        self.inputs = self.data[:, 60:]
        self.outputs = self.data[:, :60]
        self.inputs = 2 * (self.inputs - 1800) / (3100 - 1800) - 1 
        self.outputs = 2 * (self.outputs + 1000) / (2000) - 1 #scale iwth -1000 to 1000, 1800 - 3100


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
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    tot_loss = 0
    cnt = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        cnt += 1
    return tot_loss / cnt

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    cnt = 0
    pval = []
    with torch.no_grad():
        for X, y in dataloader:
            cnt += 1
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pval.append(pred.cpu().numpy())
            test_loss += loss_fn(pred, y).item()
    pval = np.concatenate(pval, axis = 0)

    print(f"Test Error: \n Avg loss: {test_loss} \n")
    return test_loss / cnt

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
    # #first 80% and last 20% split
    # training_data = TextDataset("fullmodeldata/fullrandcont.txt")
    # test_data = TextDataset("fullmodeldata/temp.txt")

    #random split 80/20
    datas = TextDataset("fullmodeldata/fullrandcont.txt")
    trainsiz = int(0.8 * len(datas))
    training_data, test_data = random_split(datas, [trainsiz, len(datas) - trainsiz])

    with open("fullmodeldata/train.txt", "w") as f:
        for i in training_data.indices: 
            data_row = datas.data[i].tolist()
            f.write(" ".join(map(str, data_row)) + "\n")
    with open("fullmodeldata/test.txt", "w") as f:
        for i in test_data.indices:
            data_row = datas.data[i].tolist()
            f.write(" ".join(map(str, data_row)) + "\n")

    train_dataloader = DataLoader(training_data, batch_size=100)
    test_dataloader = DataLoader(test_data, batch_size=100)

    train_error = []
    test_error = []

    epochs = 3000
    for t in range(epochs):
        lc = train(train_dataloader, model, loss_fn, optimizer)
        if (t % 50 == 0): 
            test_error.append(test(test_dataloader, model, loss_fn))
            train_error.append(lc)

  
    t = np.zeros(60)
    for i in range(60): t[i] = i

    plt.plot(t, test_error, label = "test error", linestyle="-.")
    plt.plot(t, train_error, label = "training loss", linestyle="-")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "fullmodelrs.pth") 

    
