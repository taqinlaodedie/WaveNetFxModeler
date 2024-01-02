import argparse
import pickle
from scipy.io import wavfile
import numpy as np

def wav_read(filename, normalize=True):
    rate, signal = wavfile.read(filename)
    signal = signal.astype(np.float32)

    if normalize == False:
        return signal
    
    # Normalize signal to a range -1 : 1
    signal = (signal - np.mean(signal)) * 2 / (signal.max() - signal.min())

    return signal

def main(args):
    in_data = wav_read(args.in_file)
    out_data = wav_read(args.out_file)
    assert len(in_data) == len(out_data), "input file has {} samples, but target file has {} samples".format(len(in_data), len(out_data))

    split = lambda d: np.split(d, [int(len(d) * 0.8)])

    d = {}
    d["x_train"], d["x_valid"] = split(in_data)
    d["y_train"], d["y_valid"] = split(out_data)

    pickle.dump(d, open(args.data, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")

    parser.add_argument("--data", default="data.pickle")
    args = parser.parse_args()
    main(args)