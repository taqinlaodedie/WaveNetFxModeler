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
import json

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y))

input_file = "ht1-input.wav"
target_file = "ht1-target.wav"
sample_size = 4410
batch_size = 10
learning_rate = 0.004
loss_function = RMSELoss()
residual_channels = 16
conv_layers = 8
repeates = 3

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

class Net(LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.wavenet = WaveNet(
            residual_channels, conv_layers, repeates,
            kernel_size=3
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
        loss = loss_function(y[:, :, -y_pred.size(2) :], y_pred).mean()
        self.log('Train loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = loss_function(y[:, :, -y_pred.size(2) :], y_pred).mean()
        self.log('Val loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return self(batch)
    
def save_model(model):
    data_out = {"type": "WaveNet",
                "residual_channels": residual_channels,
                "conv_layers": conv_layers,
                "repeats": repeates, 
                "variables": []}
    wavenet = model.wavenet

    for params in wavenet.input_conv.state_dict():
        print(params, '\t', wavenet.input_conv.state_dict()[params].size())
        data_out["variables"].append({"block": "input_conv",
                                      "name": params,
                                      "data": wavenet.input_conv.state_dict()[params].flatten().cpu().numpy().tolist()})
        
    for i in range(conv_layers):
        conv_layer_out = {"block": "causal_conv." + str(i),
                          "variables": []}
        for params in wavenet.conv_layers[i].conv.state_dict():
            print("causal_conv." + str(i) + ".conv.", params, '\t', wavenet.conv_layers[i].conv.state_dict()[params].size())
            conv_layer_out["variables"].append({"layer": "conv",
                                                "name": params,
                                                "data": wavenet.conv_layers[i].conv.state_dict()[params].flatten().cpu().numpy().tolist()})
        for params in wavenet.conv_layers[i].skip_conv.state_dict():
            print("causal_conv." + str(i) + ".skip_conv.", params, '\t', wavenet.conv_layers[i].skip_conv.state_dict()[params].size())
            conv_layer_out["variables"].append({"layer": "skip_conv",
                                                "name": params,
                                                "data": wavenet.conv_layers[i].skip_conv.state_dict()[params].flatten().cpu().numpy().tolist()})
        for params in wavenet.conv_layers[i].out_conv.state_dict():
            print("causal_conv." + str(i) + ".out_conv.", params, '\t', wavenet.conv_layers[i].out_conv.state_dict()[params].size())
            conv_layer_out["variables"].append({"layer": "out_conv",
                                                "name": params,
                                                "data": wavenet.conv_layers[i].out_conv.state_dict()[params].flatten().cpu().numpy().tolist()})
        data_out["variables"].append(conv_layer_out)

    for params in wavenet.post_conv.state_dict():
        print(params, '\t', wavenet.post_conv.state_dict()[params].size())
        data_out["variables"].append({"block": "post_conv",
                                      "name": params,
                                      "data": wavenet.post_conv.state_dict()[params].flatten().cpu().numpy().tolist()})
        
    with open('converted_model.json', 'w') as outfile:
        json.dump(data_out, outfile)
    
if __name__ == "__main__":
    model = Net()
    trainer = Trainer(
        max_epochs=100
    )
    trainer.fit(model)

    # Use this to predict
    # model = Net.load_from_checkpoint('ht1.ckpt')
    # model.eval()
    # input_data = wav_read(input_file)
    # target_data = wav_read(target_file)
    # start_index = int(len(input_data) * 0.9)
    # input_data = input_data[start_index:]
    # target_data = target_data[start_index:]
    # input_len = len(input_data) - len(input_data) % sample_size
    # input_data = input_data[:input_len].reshape(-1, 1, sample_size)
    # target_data = target_data[:input_len]
    # x = torch.from_numpy(input_data).to("cuda:0")
    # pred = torch.empty_like(x)
    # for i in range(math.ceil(x.shape[0] / batch_size)):
    #     x_batch = x[i*batch_size : (i+1)*batch_size, :, :]
    #     with torch.no_grad():
    #         pred_batch = model(x_batch)
    #     pred[i*batch_size : (i+1)*batch_size, :, :] = pred_batch
    # pred = pred.permute(0, 2, 1)
    # pred = torch.flatten(pred).detach().cpu().numpy()
    # wav_write("pred.wav", pred)
    # wav_write("target.wav", target_data)
    # plt.figure(1)
    # plt.plot(target_data, label="target")
    # plt.plot(pred, label="pred")
    # plt.legend()
    # plt.show()

    save_model(model)