from typing import Any
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from WaveNet import WaveNet
import numpy as np
from scipy.io import wavfile
import sys
import matplotlib.pyplot as plt
import math

input_file = "ht1-input.wav"
target_file = "ht1-target.wav"
sample_size = 4410
batch_size = 10
learning_rate = 1e-3
loss_function = torch.nn.MSELoss()

def wav_read(filename, normalize=True):
    rate, signal = wavfile.read(filename)
    signal = signal.astype(np.float32)

    if normalize == False:
        return signal
    
    # Normalize signal to a range -1 : 1
    signal = (signal - np.mean(signal)) * 2 / (signal.max() - signal.min())

    return signal

def wav_write(filename, signal):
    wavfile.write(filename, 44100, signal)

def load_data():
    input_raw = wav_read(input_file)
    target_raw = wav_read(target_file)
    if len(input_raw) != len(target_raw):
        print("input file has {} samples, but target file has {} samples".format(len(input_raw), len(target_raw)))
        sys.exit()
    plt.figure(0)
    plt.plot(target_raw[44100:88200], label="target")
    plt.plot(input_raw[44100:88200], label="input")
    plt.legend()
    plt.show()

    length = len(input_raw) - len(input_raw) % sample_size
    x = input_raw[:length].reshape(-1, 1, sample_size)
    y = target_raw[:length].reshape(-1, 1, sample_size)
    split = lambda d: np.split(d, [int(len(d) * 0.8)])
    d = {}
    d["x_train"], d["x_valid"] = split(x)
    d["y_train"], d["y_valid"] = split(y)

    return d 

def pre_emphasis_filter(x, coeff=0.85):
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)

def ESRLoss(y, y_pred):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)

class Net(LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.wavenet = WaveNet(
            16, 8, 3,
            kernel_size=2
        )

    def prepare_data(self):
        ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        data = load_data()
        self.train_ds = ds(data["x_train"], data["y_train"])
        self.valid_ds = ds(data["x_valid"], data["y_valid"])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.wavenet.parameters(), lr=learning_rate
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=batch_size,
            num_workers=4,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, 
            batch_size=batch_size, 
            num_workers=4,
            persistent_workers=True
        )

    def forward(self, x):
        return self.wavenet(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        # loss = ESRLoss(y[:, :, -y_pred.size(2) :], y_pred).mean()
        loss = loss_function(y[:, :, -y_pred.size(2) :], y_pred).mean()
        self.log('Train loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        # loss = ESRLoss(y[:, :, -y_pred.size(2) :], y_pred).mean()
        loss = loss_function(y[:, :, -y_pred.size(2) :], y_pred).mean()
        self.log('Val loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return self(batch)
    
if __name__ == "__main__":
    # model = Net()
    # trainer = Trainer(
    #     max_epochs=100
    # )
    # trainer.fit(model)

    # Use this to predict
    model = Net.load_from_checkpoint('ht1.ckpt')
    model.eval()
    input_data = wav_read(input_file)
    target_data = wav_read(target_file)
    start_index = int(len(input_data) * 0.9)
    input_data = input_data[start_index:]
    target_data = target_data[start_index:]
    input_len = len(input_data) - len(input_data) % sample_size
    input_data = input_data[:input_len].reshape(-1, 1, sample_size)
    target_data = target_data[:input_len]
    x = torch.from_numpy(input_data).to("cuda:0")
    pred = torch.empty_like(x)
    for i in range(math.ceil(x.shape[0] / batch_size)):
        x_batch = x[i*batch_size : (i+1)*batch_size, :, :]
        with torch.no_grad():
            pred_batch = model(x_batch)
        pred[i*batch_size : (i+1)*batch_size, :, :] = pred_batch
    pred = pred.permute(0, 2, 1)
    pred = torch.flatten(pred).detach().cpu().numpy()
    wav_write("pred.wav", pred)
    wav_write("target.wav", target_data)
    plt.figure(1)
    plt.plot(target_data, label="target")
    plt.plot(pred, label="pred")
    plt.legend()
    plt.show()