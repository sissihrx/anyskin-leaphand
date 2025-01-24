import torch
import numpy as np
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
        self.data = torch.tensor(self.data)  # Convert to PyTorch tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single sample (input, output)
        sample = self.data[idx]
        inputs = sample[-16:]  
        output = sample[:60]  
        inputs = inputs.float()
        output = output.float()

        return 2 * (inputs - min(inputs)) / (max(inputs) - min(inputs)) - 1, 2 * (output - min(output)) / (max(output) - min(output)) - 1


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 128),
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")



if __name__ == "__main__":
    training_data = TextDataset("/Users/sissi/Downloads/2/data/randomsample.txt")
    test_data = TextDataset("/Users/sissi/Downloads/2/data/data_2025-01-17_16-48-54.txt")

    train_dataloader = DataLoader(training_data, batch_size=10)
    test_dataloader = DataLoader(test_data, batch_size=10)

    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    