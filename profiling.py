import torch
from thop import profile
import argparse

def main(args):
    model = torch.load(args.path).to("cpu")
    input = torch.rand(1, 1, 1)
    macs, params = profile(model, inputs=(input, ))
    print("Model has {} params, needs {} MACs for an input of {}".format(params, macs, input.shape))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="model.pth")
    args = parser.parse_args()
    main(args)