import torch
from torch.utils.data import TensorDataset
import math
import pickle
import argparse

from WaveNet import WaveNet

class FxDataset(TensorDataset):
    def __init__(self, x, y, window_len, batch_size):
        super(TensorDataset, self).__init__()
        len_index = len(x) - (len(x) % (window_len * batch_size))
        self.x = x[:len_index].reshape(-1, window_len, 1)
        self.y = y[:len_index].reshape(-1, window_len, 1)
        self.batch_size = batch_size

    def __getitem__(self, index):
        x_out = self.x[index*self.batch_size : (index+1)*self.batch_size]
        y_out = self.y[index*self.batch_size : (index+1)*self.batch_size]
        return x_out, y_out
    
    def __len__(self):
        return math.floor(len(self.x) / self.batch_size)
    
    def shuffle(self):
        index = torch.randperm(len(self.x))
        self.x = self.x[index]
        self.y = self.y[index]

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y))
    
def train_epoch(model:WaveNet, dataset:FxDataset, loss_function, optimizer:torch.optim.Optimizer, up_fr=512):
    epoch_loss = 0
    dataset.shuffle()
    for i in range(len(dataset)):
        x, y = dataset[i]
        optimizer.zero_grad()
        y_pred = model.forward(x)
        loss = loss_function(y, y_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss
        print("Batch {}/{}: {:.2%}".format(i, len(dataset), i/len(dataset)), end='\r')
    epoch_loss = epoch_loss / (i + 1)
    return epoch_loss

def valid_epoch(model:WaveNet, dataset:FxDataset, loss_function):
    with torch.no_grad():
        epoch_loss = 0
        for i in range(len(dataset)):
            x, y = dataset[i]
            y_pred = model.forward(x)
            loss = loss_function(y, y_pred)
            epoch_loss += loss
        epoch_loss = epoch_loss / (i + 1)
    return epoch_loss

def main(args):
    model = WaveNet(
        num_channels=args.num_channels,
        num_layers=args.num_layers,
        num_repeats=args.num_repeats,
        kernel_size=3
    )

    if torch.cuda.is_available():
        print("Use GPU to train")
        device = "cuda:0"
    else:
        print("Use CPU to train")
        device = "cpu"
    model.to(device)

    ds = lambda x, y: FxDataset(
        torch.from_numpy(x).to(device), 
        torch.from_numpy(y).to(device), 
        window_len=args.sequence_length, 
        batch_size=args.batch_size
    )
    data = pickle.load(open(args.data, "rb"))
    train_dataset = ds(data["x_train"], data["y_train"])
    valid_dataset = ds(data["x_valid"], data["y_valid"])

    loss_function = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = 1e6

    for i in range(args.max_epochs):
        print("*********** Epoch {} **************".format(i+1))
        train_loss = train_epoch(model, train_dataset, loss_function, optimizer)
        valid_loss = valid_epoch(model, valid_dataset, loss_function)
        print("Train loss {}, Valid loss {}".format(train_loss, valid_loss))
        if valid_loss < best_loss:
            torch.save(model, "model.pth")
            best_loss = valid_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channels", type=int, default=16)
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=8)

    parser.add_argument("--sequence_length", type=int, default=4410)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--max_epochs", type=int, default=1500)

    parser.add_argument("--data", default="data.pickle")
    args = parser.parse_args()
    main(args)