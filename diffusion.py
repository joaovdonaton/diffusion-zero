import torch
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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


class TrainDiffussionCFG:
    def __init__(self, score_network, dl, schedule, eta):
        """
        Args:
        - score_network: unet.UNet instance that will be updated
        - dl: dataloader for the dataset
        - eta: probability of dropping label and doing unconditional matching (need for classifier-free guidance)
        """
        self.eta = eta
        self.loader = dl
        self.net = score_network
        self.schedule = schedule
    
    def train(self, epoch_count):
        optimizer = optim.AdamW(self.net.parameters(), lr=0.0001)
        self.net.train()

        for i in range(epoch_count):
            print(f'Epoch {i} ----------')
            losses = []

            for z, y in self.loader:
                z = z.to(device) # sample from our target distribution
                y = y.to(device=device, dtype=torch.float32) # label for that sample

                # drop labels for samples with probability self.eta
                y[torch.rand_like(y) < self.eta] = 10 # label 10 (i.e 11th label) is for unconditional

                t = torch.rand_like(y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                noise = torch.randn_like(z)

                # why didnt t work without unsqueeze, shouldnt it be broadcasting? i.e (64) * (64,1,32,32)

                x_t = self.schedule.alpha(t)*z + self.schedule.beta(t)*noise

                # for our loss function we will do L = ||score_network(x_t) - noise||^2
                loss = ((self.net(x_t, t, y) - noise)**2).mean()

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            print(f'Loss at the end of epoch {i}: {sum(losses)/len(losses)}')

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
        self.network.eval()

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
    # sig = DiffusionCoefficient(1)
    # sim = SimulateDiff(unet, sig)

    #print(sim.simulate(0, 100))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]) # means and stds from data.ipynb for cifar 10 channels
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=500, shuffle=True, num_workers=2)

    ns = NoiseScheduler()
    train = TrainDiffussionCFG(unet, train_loader, ns, 0.1)
    train.train(100)