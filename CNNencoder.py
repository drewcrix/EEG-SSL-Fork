import numpy as np
import torch
import torch.nn as nn
import warnings
from matplotlib import pyplot as plt
from moabb.datasets import PhysionetMI
from typing import Sequence, Union

warnings.filterwarnings("ignore")


dataset = PhysionetMI()
subject = 1
run = 4


raw = dataset._load_one_run(subject, run)

print(raw.get_data().shape)
print("Raw info:")

data = raw.get_data() #this has shape (64, 20000) for 64 channels and 2minutes of data at 160Hz sampling rate


print("Data shape:", data.shape) #data shape (channels, timepoints)
time_elapsed = 1/raw.info['sfreq'] * data.shape[1] #in seconds
print("Total time elasped in this test (s)", time_elapsed) 

f = data[0]  # First channel
y = np.arange(0, time_elapsed, 1/raw.info['sfreq'])  # Time vector



plt.plot(y, f) #plotting the EEG signal from the first channel to visualize the input data for the CNN layer
plt.title('EEG Signal from First Channel')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.show()



class Block(nn.Module):
    """
    EEGNet-like temporal block. Followed similar format but removed spatial filer from depthwise conv2d
        Conv1d -> BatchNorm1d -> PReLU -> AvgPool1d -> Dropout
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, pool_size: int, dropout: float = 0.5, stride: int = 1, padding: Union[int, str] = "same"):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.PReLU()
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x

#CNN Encoder. Included the temporal block above to make this clearer. Can adjust the amount of temporal blocks and parameters
class CNNEncoder(nn.Module):
    def __init__(
        self,
        output_channels: int = 4,
        kernel_sizes: Sequence[int] = (128, 64, 32), #Compare these sizes with sfreq to make sure the sizes are ok 
        pool_sizes: Sequence[int] = (5, 3, 2),
        dropout: float = 0.5,
        stride: int = 1,
        padding: Union[int, str] = "same"
    ):
        super().__init__()

        if len(kernel_sizes) != 3:
            raise ValueError("kernel_sizes must have length 3")
        if len(pool_sizes) != len(kernel_sizes):
            raise ValueError("length of pool vector and kernel vector must match")


        o1 = output_channels
        o2 = o1 * 2
        o3 = o2 * 2

        self.block1 = Block(1,  o1, kernel_sizes[0], pool_sizes[0], dropout, stride, padding)
        self.block2 = Block(o1, o2, kernel_sizes[1], pool_sizes[1], dropout, stride, padding)
        self.block3 = Block(o2, o3, kernel_sizes[2], pool_sizes[2], dropout, stride, padding)

        self.F = o3  # output features per electrode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape

        # Done to prevent channel mixing and spatial fileters. 
        # Process each electrode independently: (B, C, T) -> (B*C, 1, T)
        x = x.reshape(B * C, 1, T)

        # Temporal CNN blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

       # x = x.mean(dim=-1)   #averages the strength of the features over time.

        _, F, Tp = x.shape()

        # Back to per-electrode node embeddings: (B, C, F)
        x = x.reshape(B, C, F, Tp) #keeping the time dimension 
        return x

model = CNNEncoder()


