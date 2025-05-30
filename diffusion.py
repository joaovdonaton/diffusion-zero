import torch
import unet
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Note that all terminology and formulas used here are defined in the accompanying readme and notes

# Noise scheduler for our probability path definition:
# p_t(x_t|z) = N(alpha(t)*z, beta(t)^2*I_d) 
# where z is a sample from target distribution and N(mean, std) is notation for normal distribution
# We will use alpha(t)=t and beta(t)=1-t
# In other words, we start at N(0, I_d) (noise) and end up at a N(z, 0) (deterministic z)
class NoiseScheduler:
    def __init__(self):
        pass

    # for all of these, take t of shape (BATCH, )

    def alpha(self, t):
        return t

    def beta(self, t):
        return torch.ones_like(t)-t

    def alpha_dt(self, t):
        return torch.ones_like(t)
    
    def beta_dt(self, t):
        return torch.ones_like(t)*(-1)


# just constant at every t for now
class DiffusionCoefficient:
    def __init__(self, value):
        self.coef = value
    
    def __call__(self, t):
        """
        Args:
        - t: (batch,)
        """
        return self.coef*torch.ones_like(t)


class TrainDiff:
    def __init__(self):
        pass
    
    def train(self):
        pass


# We'll use Euler Maruyama method to simulate our SDE for inference
class SimulateDiff:
    def __init__(self, score_network, diff_coef):
        """
        Args:
        - score_network: trained unet.UNet instance
        """
        self.network = score_network
        self.sigma = diff_coef
    
    @torch.no_grad
    def simulate(self, y_label, timesteps):
        """
        Args:
        - y_label: class label for CFG
        - timesteps
        """
        step_size = 1/timesteps
        x = torch.randn(1, 1, 32, 32).to(device) # (bs, c, h, w)
        t = torch.zeros(1, 1, 1, 1).to(device)
        y = torch.tensor(y_label).to(device)
        for _ in range(timesteps):
            noise = torch.randn(1, 1, 32, 32).to(device)

            x = x + step_size*self.network(x, t, y) + self.sigma(t)*math.sqrt(step_size)*noise
            t = t + step_size

        return x 


if __name__ == '__main__':
    unet = unet.UNet([2, 4, 8], 2, 32, 10).to(device)
    sig = DiffusionCoefficient(1)
    sim = SimulateDiff(unet, sig)

    print(sim.simulate(0, 100))