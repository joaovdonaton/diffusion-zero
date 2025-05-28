import torch
from torch import nn
import math

# Implementation of the neural network architecture that we need to learn the score function
# We'll use a standard UNet from Peter's course (see readme)
# Note that this is implemented in lab 3 for the course, the code here is in part copied from that with my own notes added

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


# Preserves channels #, and dimensionality, injects data from time and class embeddings and runs through some convolutions
class ResidualLayer(nn.Module):
    def __init__(self, channels, time_embed_dim, y_embed_dim):
        super().__init__()

        # BatchNorm2d applies operation described here https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html channel-wise.
        # NOTE: I'm not entirely sure why Peter's course applies the activation first, try changing this around...

        self.block1 = nn.Sequential(
            nn.SiLU(), 
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

        # the MLPs that will process time and class information to be added onto our channels
        # looks like model will learn to take the embeddings, and add relevant per-channel data (notice how the output dim is channel size)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels)
        )
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, channels)
        )

    def forward(self, x, temb, yemb):
        """
        Args:
        - x: (bs, c, h, w)
        - temb: (bs, t_embed_dim)
        - yemb: (bs, y_embed_dim)
        """
        res = x.clone()

        x = self.block1(x)

        t_data = self.time_adapter(temb).unsqueeze(-1).unsqueeze(-1) # shape (bs, c, 1, 1)
        x = x + t_data # (bs, c, h, w) + (bs, c, 1, 1) will do broadcast 

        y_data = self.y_adapter(yemb).unsqueeze(-1).unsqueeze(-1)
        x = x + y_data

        x = self.block2(x)

        x = x+res 
        return x


# N residual layers followed by a downsampling convolution layer
class Encoder(nn.Module):
    def __init__(self, c_in, c_out, num_residual_layers, t_embed_dim, y_embed_dim):
        """
        Args: All integers
        """
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(c_in, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)]
        )

        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x, temb, yemb):
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        for b in self.res_blocks:
            x = b.forward(x, temb, yemb)

        x = self.downsample(x)
        
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    # temb = TimeEmbedder(16)
    # print(temb.forward(torch.randn(4, 1)))
    
    x = torch.randn(1, 4, 10, 10)
    temb = TimeEmbedder(32)
    yemb = torch.randn(1, 10)

    # res = ResidualLayer(4, 32, 10)
    # y = res.forward(x, temb.forward(torch.randn(1, 1)), yemb)
    # print(y)
    # print(y.shape)

    # enc = Encoder(4, 6, 2, 32, 10)
    # y = enc.forward(x, temb.forward(torch.randn(1,1)), yemb)
    # print(y)
    # print(y.shape)