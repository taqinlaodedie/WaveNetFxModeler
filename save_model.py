import argparse
import numpy as np
import json
import torch

def save_model(args):
    model = torch.load(args.model)

    data_out = {"type": "WaveNet", 
                "num_channels": args.num_channels,
                "num_layers": args.num_layers,
                "num_repeats": args.num_repeats,
                "kernel_size": 3,
                "variables": []}
    
    for params in model.input_conv.state_dict():
        print(params, '\t', model.input_conv.state_dict()[params].size())
        data_out["variables"].append({"block": "input_conv",
                                      "name": params,
                                      "data": model.input_conv.state_dict()[params].flatten().cpu().numpy().tolist()})
        
    for i in range(args.num_layers):
        conv_layer_out = {"block": "causal_conv." + str(i),
                          "variables": []}
        for params in model.conv_layers[i].conv.state_dict():
            print("causal_conv." + str(i) + ".conv.", params, '\t', model.conv_layers[i].conv.state_dict()[params].size())
            conv_layer_out["variables"].append({"layer": "conv",
                                                "name": params,
                                                "data": model.conv_layers[i].conv.state_dict()[params].flatten().cpu().numpy().tolist()})
        for params in model.conv_layers[i].skip_conv.state_dict():
            print("causal_conv." + str(i) + ".skip_conv.", params, '\t', model.conv_layers[i].skip_conv.state_dict()[params].size())
            conv_layer_out["variables"].append({"layer": "skip_conv",
                                                "name": params,
                                                "data": model.conv_layers[i].skip_conv.state_dict()[params].flatten().cpu().numpy().tolist()})
        for params in model.conv_layers[i].out_conv.state_dict():
            print("causal_conv." + str(i) + ".out_conv.", params, '\t', model.conv_layers[i].out_conv.state_dict()[params].size())
            conv_layer_out["variables"].append({"layer": "out_conv",
                                                "name": params,
                                                "data": model.conv_layers[i].out_conv.state_dict()[params].flatten().cpu().numpy().tolist()})
        data_out["variables"].append(conv_layer_out)

    for params in model.post_conv.state_dict():
        print(params, '\t', model.post_conv.state_dict()[params].size())
        data_out["variables"].append({"block": "post_conv",
                                      "name": params,
                                      "data": model.post_conv.state_dict()[params].flatten().cpu().numpy().tolist()})
        
    with open('converted_model.json', 'w') as outfile:
        json.dump(data_out, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--num_channels", type=int, default=16)
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--name", default="converted_model")
    args = parser.parse_args()
    save_model(args)