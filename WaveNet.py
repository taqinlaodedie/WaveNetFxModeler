"""
Implementation of the modified WaveNet in article https://arxiv.org/pdf/1811.00334.pdf
Original WaveNet referenced by https://arxiv.org/pdf/1609.03499.pdf
"""
import torch
import torch.nn as nn

class CausalConv(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv, self).__init__(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.__padding)

    def forward(self, x):
        y = super(CausalConv, self).forward(x)

        if self.__padding != 0:
            y = y[:, :, :-self.__padding]
        return y
    
class WaveNet(nn.Module):
    def __init__(self, num_channels, num_layers, num_repeats, kernel_size=2):
        super(WaveNet, self).__init__()
        self.num_channels = num_channels

        dilations = [2 ** d for d in range(num_layers)] * num_repeats
        gated_channels = 2 * num_channels

        self.input_conv = nn.Conv1d(1, num_channels, kernel_size=1)
        self.dilated_convs = nn.ModuleList([
            CausalConv(num_channels, gated_channels, kernel_size, d)
            for i, d in enumerate(dilations)
        ])
        self.residual_convs = nn.ModuleList([
            nn.Conv1d(num_channels, num_channels, kernel_size=1)
            for i, d in enumerate(dilations)
        ])
        self.post_conv = nn.Conv1d(num_channels * num_layers * num_repeats, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        output = x
        output = self.input_conv(output)
        
        # Gather all skip outputs and discard the last residual output
        for dilated_conv, residual_conv in zip(self.dilated_convs, self.residual_convs):
            x = output
            dilated_conv_out = dilated_conv(output)
            splited_dilated_conv_out = torch.split(dilated_conv_out, self.num_channels, dim=1)
            skip = torch.tanh(splited_dilated_conv_out[0]) * torch.sigmoid(splited_dilated_conv_out[1])
            skips.append(skip)
            output = residual_conv(skip)
            output = output + x[:, :, -output.size(2) :]

        output = torch.cat([s[:, :, -output.size(2):] for s in skips], dim=1)
        output = self.post_conv(output)
        return output