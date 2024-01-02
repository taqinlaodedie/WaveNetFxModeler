"""
Implementation of the modified WaveNet in article https://arxiv.org/pdf/1811.00334.pdf
Original WaveNet referenced by https://arxiv.org/pdf/1609.03499.pdf
"""
import torch
import torch.nn as nn

class CausalConv(nn.Module):
    def __init__(self, residual_channels, gated_channels, kernel_size, dilation=1):
        super(CausalConv, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.residual_channels = residual_channels
        self.conv = nn.Conv1d(residual_channels, gated_channels, 
                              kernel_size=kernel_size, padding=padding, dilation=dilation)
        gate_out_channels = gated_channels // 2
        self.skip_conv = nn.Conv1d(gate_out_channels, residual_channels, 
                                     kernel_size=1, dilation=1)
        self.out_conv = nn.Conv1d(gate_out_channels, residual_channels, 
                                     kernel_size=1, dilation=1)
        
    def forward(self, x):
        residual = x

        x = self.conv(x)
        x = x[:, :, :residual.size(2)]

        a, b = torch.split(x, self.residual_channels, dim=1)
        x = torch.tanh(a) * torch.sigmoid(b)
        
        s = self.skip_conv(x)
        x = self.out_conv(s)

        x = x + residual[:, :, -x.size(2) :]

        return x, s
    
class WaveNet(nn.Module):
    def __init__(self, num_channels, num_layers, num_repeats, kernel_size=3):
        super(WaveNet, self).__init__()
        self.num_channels = num_channels

        dilations = [2 ** d for d in range(num_layers)] * num_repeats
        gated_channels = 2 * num_channels

        self.input_conv = nn.Conv1d(1, num_channels, kernel_size=1)
        self.conv_layers = nn.ModuleList([
            CausalConv(num_channels, gated_channels, kernel_size, d)
            for i, d in enumerate(dilations)
        ])
        self.post_conv = nn.Conv1d(num_channels * num_layers * num_repeats, 1, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        skips = []
        output = x
        output = self.input_conv(output)
        
        # Gather all skip outputs and discard the last residual output
        for f in self.conv_layers:
            x = output
            output, skip = f(x)
            skips.append(skip)

        output = torch.cat([s[:, :, -output.size(2):] for s in skips], dim=1)
        output = self.post_conv(output)
        output = output.permute(0, 2, 1)
        return output
