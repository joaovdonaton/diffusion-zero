import torch

# Note that all terminology and formulas used here are defined in the accompanying readme and notes

# Noise scheduler for our probability path definition:
# p_t(x_t|z) = N(alpha(t)*z, beta(t)^2*I_d) 
# where z is a sample from target distribution and N(mean, std) is notation for normal distribution
# We will use alpha(t)=t and beta(t)=1-t
# In other words, we start at N(0, I_d) (noise) and end up at a N(z, 0) (deterministic z)
class NoiseScheduler:
    def __init__(self):
        pass

    # for all of these, take t of shape (BATCH, DIMENSION)

    def alpha(self, t):
        return t

    def beta(self, t):
        return torch.ones_like(t)-t

    def alpha_dt(self, t):
        return torch.ones_like(t)
    
    def beta_dt(self, t):
        return torch.ones_like(t)*(-1)