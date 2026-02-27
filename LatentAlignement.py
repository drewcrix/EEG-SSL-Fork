import torch
import torch.nn as nn
from collections import OrderedDict

class LatentProjector(nn.Module): 
    def __init__(self, z_dim: int, outputdim: int, hdim: int, hdim2: int): 
        super().__init__()

        self.net = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(z_dim, hdim)),
                ("activation1", nn.PReLU()),
                ("fc2", nn.Linear(hdim, hdim2)),
                ("activation2", nn.PReLU()),
                ("fc3", nn.Linear(hdim2, outputdim),
            ])
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor: #I need to pass here the concatinated vector along the batch dimension into the latent alignment block. 
        return self.net(z)

    


