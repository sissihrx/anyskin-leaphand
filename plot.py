import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, file_path, delimiter=" "):
        self.data = []
        with open(file_path, "r") as file:
            for line in file:
                self.data.append([float(x) for x in line.strip().split(delimiter)])
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        self.inputs = self.data[:, -16:]
        self.outputs = self.data[:, :60]
        self.inputs = 2 * (self.inputs - self.inputs.min()) / (self.inputs.max() - self.inputs.min()) - 1
        self.outputs = 2 * (self.outputs - self.outputs.min()) / (self.outputs.max() - self.outputs.min()) - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


# Define model
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss =  0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches

    print(f"Test Error: \n Avg loss: {test_loss} \n")
    return pred.cpu().numpy()

def gety(dataloader):
    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(device)
    return y.cpu().numpy()



if __name__ == "__main__":
    training_data = TextDataset("/Users/sissi/Downloads/pythonstuff/anyskin/anyskin/visualizations/data_2025-02-03_20-39-21.txt")
    test_data = TextDataset("/Users/sissi/Downloads/pythonstuff/anyskin/anyskin/visualizations/contdata_2025-02-06_11-55-46.txt")

    train_dataloader = DataLoader(training_data, batch_size=30)
    test_dataloader = DataLoader(test_data, batch_size=100)

    epochs = 300
    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        if (t % 50 == 0): test(test_dataloader, model, loss_fn)

   
    res = test(test_dataloader, model, loss_fn).T
    act = gety(test_dataloader).T
    t = np.zeros(len(res[0]))
    for i in range(len(res[0])):
        t[i] = i
    plt.plot(t, res[50], label = "predicted", linestyle="-.")
    plt.plot(t, act[50], label = "actual", linestyle="-")
    plt.legend()
    plt.show()



    print("Done!")
    
