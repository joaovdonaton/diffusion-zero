import torch
from torch import nn
import math

# Implementation of the neural network architecture that we need to learn the score function
# We'll use a standard UNet from Peter's course (see readme)

class TimeEmbedder(nn.Module):
    def __init__(self, dim):
        """
        dim should be %2 = 0
        """
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dim//2))
    
    def forward(self, t):
        """
        We are making embedding using [sin(2pi*w_1*t), ..., cos(2pi*w_1*t), ...]

        Args:
            t of shape (batch_size, 1)

        Return:
            time embedding shape (batch_size, dim)
        """
        angles = self.get_angles(t) # shapes are (1, dim//2) * (bs, 1) so we get (bs, dim//2) bc of broadcasting
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)

        # Also, we want to *sqrt(2) because the embedding will have variance 1/2
        # To keep consistency of having data mean=0 and variance=1, we multiply by sqrt2 to get variance 1
        # Basically: var(x)=0.5, so var(sqrt(2)*x)=2*var(x)=2*0.5=1
        return torch.cat([sin_emb, cos_emb], dim=1) * math.sqrt(2)

    # debug    
    def get_angles(self, t):
        return 2 * math.pi * self.weights * t

class UNet(nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    temb = TimeEmbedder(16)
    print(temb.forward(torch.randn(4, 1)))