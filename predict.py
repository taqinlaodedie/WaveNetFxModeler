import torch
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal.windows import hamming

from train import FxDataset

def wav_write(filename, signal):
    wavfile.write(filename, 44100, signal.astype(np.float32))

def main(args):
    model = torch.load(args.path)
    data = pickle.load(open(args.data, "rb"))
    
    device = "cpu"
    model.to(device)
    dataset = FxDataset(
        torch.from_numpy(data["x_valid"]).to(device), 
        torch.from_numpy(data["y_valid"]).to(device), 
        args.sequence_length, 
        args.batch_size
    )

    with torch.no_grad():
        frame_size = args.batch_size * args.sequence_length
        pred = np.zeros(len(dataset) * frame_size)
        target = np.zeros(len(dataset) * frame_size)
        for i in range(len(dataset)):
            x, y = dataset[i]
            y_pred = model(x)
            pred[i*frame_size : (i+1)*frame_size] = torch.flatten(y_pred).detach().to("cpu").numpy()
            target[i*frame_size : (i+1)*frame_size] = torch.flatten(y).detach().to("cpu").numpy()

    wav_write("pred.wav", pred)
    wav_write("target.wav", target)
    
    plt.figure(1)
    plt.title("Samples of first 10s")
    plt.plot(target[:441000], label='target')
    plt.plot(pred[:441000], label='pred')
    plt.legend()
    plt.show()

    N = 1024
    start_ind = 0
    pred = pred[start_ind : start_ind+int(N/4)]
    target = target[start_ind : start_ind+int(N/4)]
    pred = np.concatenate((pred, np.zeros(int(3*N/4))))
    target = np.concatenate((target, np.zeros(int(3*N/4))))
    window = hamming(N)
    pred *= window
    target *= window
    fft_pred = 20 * np.log10(np.abs(fft(pred)) / N * 2.0)
    fft_pred = fft_pred[range(int(N / 2))]
    fft_target = 20 * np.log10(np.abs(fft(target)) / N * 2.0)
    fft_target = fft_target[range(int(N / 2))]
    fre = np.arange(int(N / 2)) * 44100.0 / N

    plt.figure(2)
    plt.title("Frequency")
    plt.plot(fre, fft_target, label='target')
    plt.plot(fre, fft_pred, label='pred')
    plt.xlabel("Frequency/Hz")
    plt.ylabel("dB")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.pickle")
    parser.add_argument("--path", default="model.pth")
    parser.add_argument("--sequence_length", type=int, default=88200)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)