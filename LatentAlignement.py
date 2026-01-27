import torch
import torch.nn as nn
from collections import OrderedDict

class LatentProjector(nn.Module): 
    def __init__(self, z_dim: int, outputdim: int, hdim: int): 
        super().__init__()

        self.net = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(z_dim, hdim)),
                ("activation1", nn.PReLU()),
                ("fc2", nn.Linear(hdim, outputdim)),
                ("activation2", nn.PReLU()),
            ])
        )
    
    def forward(self, A: torch.Tensor) -> torch.Tensor: #I need to pass here the concatinated vector along the batch dimension into the latent alignment block. 
        z = torch.cat(A, dim=0) #concatenating all GNN outputs on the Batch Dimension. Pass here an array of all outputs from GNN encoder
        return self.net(z)

    


