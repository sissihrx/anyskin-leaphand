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
        
        self.inputs = self.data[:, 30:]
        self.outputs = self.data[:, :30]
        self.outputs -= self.outputs[0]
        
        print(self.outputs[0])
        print(self.outputs.max())
        
        self.inputs = 2 * (self.inputs - 1800) / (3100 - 1800) - 1 
        self.outputs = 2 * (self.outputs + 100) / 350 - 1 #scale iwth -100 to 250, 1800 - 3100
        

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.7)

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
    # datas = TextDataset("2fmodeldata/2frandcontdata.txt")
    # trainsiz = int(0.8 * len(datas))
    # training_data, test_data = random_split(datas, [trainsiz, len(datas) - trainsiz])

    # with open("2fmodeldata/train2f.txt", "w") as f:
    #     for i in training_data.indices: 
    #         data_row = datas.data[i].tolist()
    #         f.write(" ".join(map(str, data_row)) + "\n")
    # with open("test.txt", "w") as f:
    #     for i in test_data.indices:
    #         data_row = datas.data[i].tolist()
    #         f.write(" ".join(map(str, data_row)) + "\n")
    training_data = TextDataset("2fingermodel/2fmodeldata/moredata.txt")
    test_data = TextDataset("2fingermodel/2fmodeldata/moredata.txt")

    train_dataloader = DataLoader(training_data, batch_size=100)
    test_dataloader = DataLoader(test_data, batch_size=100)
    
    print(gety(test_dataloader))

    epochs = 2000
    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        if (t % 50 == 0): test(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), "2fmodelmore.pth") 

    res = test(test_dataloader, model, loss_fn).T
    act = gety(test_dataloader).T
    t = np.zeros(len(res[0]))
    for i in range(len(res[0])):
        t[i] = i
    plt.plot(t, res[3], label = "predicted", linestyle="-.")
    plt.plot(t, act[3], label = "actual", linestyle="-")
    plt.legend()
    plt.show()

    
