import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from fedrl_bench.paths import DATA_DIR

training_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

test_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

learning_rate = 1e-1
batch_size = 64
epochs = 10

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model: NeuralNetwork, loss_fn, optimiser: torch.optim.SGD):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Prediction and loss
        optimiser.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimiser.step()
        
        if batch % 100 == 0:
            current = batch * batch_size + len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

def train_loop_manual(dataloader, model: NeuralNetwork, loss_fn):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Prediction and loss
        model.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        # Manual optimiser step
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    # More accurate: p.add_(p.grad, alpha=-learning_rate)
                    p -= learning_rate * p.grad
        
        if batch % 100 == 0:
            current = batch * batch_size + len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

def step_train_optimiser(model: NeuralNetwork, loss_fn, optimiser: torch.optim.SGD, X, y, batch, size):
    model.train()
    optimiser.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    
    # Backpropagation
    loss.backward()
    optimiser.step()
    
    if batch % 100 == 0:
        current = batch * batch_size + len(X)
        print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

def step_train_manual(model: NeuralNetwork, loss_fn, X, y, batch, size):
    model.train()
    model.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    
    # Backpropagation
    loss.backward()
    # Manual optimiser step
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                # More accurate: p.add_(p.grad, alpha=-learning_rate)
                p -= learning_rate * p.grad
    
    if batch % 100 == 0:
        current = batch * batch_size + len(X)
        print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
        

def test_loop(dataloader, model: NeuralNetwork, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main() -> None:
    print("Hello from fedrl-bench!")
    device = "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    model_opt = copy.deepcopy(model)
    print(model)
    
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
    size = len(train_dataloader.dataset) # type: ignore
    # Test optimiser is equal
    for batch, (X, y) in enumerate(train_dataloader):
        step_train_optimiser(model, loss_fn, optimiser, X, y, batch, size)
        step_train_manual(model_opt, loss_fn, X, y, batch, size)
        if batch == 0:
            # Show difference caused by first calculation due to difference in FP calc accuracy
            print(f"maxdiff: {max((p-q).abs().max().item() for p,q in zip(model.parameters(), model_opt.parameters())):.3e}")
    # train_loop(train_dataloader, model, loss_fn, optimiser)
    # train_loop_manual(train_dataloader, model_opt, loss_fn)
    
    identical = all(
        torch.allclose(p, q)
        for p, q in zip(model.parameters(), model_opt.parameters(), strict=True)
    )
    print(f"parameters identical: {identical}")
    
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train_loop(train_dataloader, model, loss_fn, optimiser)
    #     test_loop(test_dataloader, model, loss_fn)
    
    # X = torch.rand(1, 28, 28, device=device)
    # print(X)
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")
    # gym.make("CartPole-v1")
    
if __name__ == "__main__":
    main()