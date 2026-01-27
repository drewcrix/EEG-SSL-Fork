import numpy as np
import torch
import torch.nn as nn
import warnings
from matplotlib import pyplot as plt
from moabb.datasets import PhysionetMI

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



class CNNEncoder1(nn.Sequential): #produces temporal feature maps from raw EEG signals
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding):
        super(CNNEncoder1, self).__init__()

        i2 = out_channels 
        o2 = i2 * 2
        k2 = kernel_size // 2

        i3 = o2
        o3 = i3 * 2
        k3 = kernel_size // 4


        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.prelu1 = nn.PReLU()
        self.AvgPool1 = nn.AvgPool1d(5, 5)
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv1d(i2, o2, k2, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(o2)
        self.prelu2 = nn.PReLU()
        self.AvgPool2 = nn.AvgPool1d(3, 3)
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Conv1d(i3, o3, k3, stride, padding, bias=False)
        self.bn3 = nn.BatchNorm1d(o3)
        self.prelu3 = nn.PReLU()
        self.AvgPool3 = nn.AvgPool1d(2, 2)
        self.dropout3 = nn.Dropout(p=0.5)

        self.F = o3  # save for later if you want

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        B, C, T = x.shape
        x = x.reshape(B*C, 1, T)


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.AvgPool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.AvgPool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        x = self.AvgPool3(x)
        x = self.dropout3(x)

        x = torch.mean(x, dim=-1)  # global average pooling over time
        x = x.reshape(B, C, self.F)  # reshape back to (B, C, F)

        return x  



## ---------------------------

## Training CNN LAYER SECTION
model1 = CNNEncoder1(in_channels=1, out_channels=4, kernel_size=128, stride=1, padding='same')



## ---------------------------

## Second CNN encoder using groups. Different encoder per channel
class CNNEncoder2(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding):
        super(CNNEncoder2, self).__init__()
        C = in_channels
        i2 = out_channels 
        o2 = i2 * 2
        k2 = kernel_size // 2

        i3 = o2
        o3 = i3 * 2
        k3 = kernel_size // 4


        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=C)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.prelu1 = nn.PReLU()
        self.AvgPool1 = nn.AvgPool1d(5, 5)
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv1d(i2, o2, k2, stride, padding, bias=False, groups=C)
        self.bn2 = nn.BatchNorm1d(o2)
        self.prelu2 = nn.PReLU()
        self.AvgPool2 = nn.AvgPool1d(3, 3)
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Conv1d(i3, o3, k3, stride, padding, bias=False, groups=C)
        self.bn3 = nn.BatchNorm1d(o3)
        self.prelu3 = nn.PReLU()
        self.AvgPool3 = nn.AvgPool1d(2, 2)
        self.dropout3 = nn.Dropout(p=0.5)

        self.F = o3 // 64 # save for later if you want
        self.Channels = C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.AvgPool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.AvgPool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        x = self.AvgPool3(x)
        x = self.dropout3(x)

        x = torch.mean(x, dim=-1)  # global average pooling over time

        B = x.shape[0] #batch size
        x = x.view(B, self.Channels, self.F) # reshape back to (B, C, F) => Input required for GCN

        return x
    
model = CNNEncoder2(in_channels=64, out_channels=256, kernel_size=128, stride=1, padding='same')