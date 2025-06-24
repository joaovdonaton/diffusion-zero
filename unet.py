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

        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1) # 2x downsample
    
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


# No down/up sampling, just residuals
class Midcoder(nn.Module):
    def __init__(self, c_in, num_residual_layers, t_embed_dim, y_embed_dim):
        """
        Args: All integers
        """
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(c_in, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)]
        )
    
    def forward(self, x, temb, yemb):
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        for b in self.res_blocks:
            x = b.forward(x, temb, yemb)

        return x


# upsampling convolutions, followed by N residual layers
class Decoder(nn.Module):
    def __init__(self, c_in, c_out, num_residual_layers, t_embed_dim, y_embed_dim):
        """
        Args: All integers
        """
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(c_out, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)]
        )

        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
    
    def forward(self, x, temb, yemb):
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        x = self.upsample(x)

        for b in self.res_blocks:
            x = b.forward(x, temb, yemb)

        return x

class UNet(nn.Module):
    def __init__(self, channels, num_residual_layers, t_embed_dim, y_embed_dim):
        """
        Args:
        - channels: list
        - others: integers
        """
        super().__init__()
        
        self.init_conv = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=3, padding=1), nn.BatchNorm2d(channels[0]), nn.SiLU())

        self.time_embedder = TimeEmbedder(t_embed_dim)

        ## CIFAR-10 has 10 classes, +1 for null class for classifier-free guidance
        # nn.Embedding is a learned lookup table, more efficient than having a linear layer because we can just lookup embeddings instead of calculating every time
        # also, encodes semantic structure in the y_embed_dim vector space as opposed to one hot encoding
        self.y_embedder = nn.Embedding(num_embeddings = 11, embedding_dim = y_embed_dim)
    
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)

        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x, t, y):
        """
        Args:
        - x: (bs, 3, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, 3, h, w)
        """

        temb = self.time_embedder.forward(t.squeeze(-1).squeeze(-1))
        yemb = self.y_embedder(y.to(dtype=torch.long))

        x = self.init_conv(x)

        residuals = []

        for enc in self.encoders:
            x = enc.forward(x, temb, yemb)
            residuals.append(x.clone())

        x = self.midcoder.forward(x, temb, yemb)

        for dec in self.decoders:
            res = residuals.pop()
            x = x + res
            x = dec(x, temb, yemb)

        x = self.final_conv(x)

        return x

if __name__ == "__main__":
    # temb = TimeEmbedder(16)
    # print(temb.forward(torch.randn(4, 1)))
    
    x = torch.randn(1, 1, 32, 32)
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
    
    # mid = Midcoder(4, 3, 32, 10)
    # y = mid.forward(x, temb.forward(torch.randn(1,1)), yemb)
    # print(y)
    # print(y.shape)
    
    # dec = Decoder(4, 2, 2, 32, 10)
    # y = dec.forward(x, temb.forward(torch.randn(1,1)), yemb)
    # print(y)
    # print(y.shape)

    unet = UNet([2, 4, 8], 2, 32, 10)
    y = unet(x, torch.randn(1, 1, 1, 1), torch.randint(11, (1,)))
    print(y)
    print(y.shape)