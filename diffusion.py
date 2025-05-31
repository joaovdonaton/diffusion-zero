import torch
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import unet
import math
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Note that all terminology and formulas used here are defined in the accompanying readme and notes

# according to formula from song's paper
class DiffusionCoefficient:
    def __init__(self):
        self.sigma_min = 0.01
        self.sigma_max = 50
    
    def __call__(self, t):
        """
        Args:
        - t: (batch,)
        """
        return self.sigma_min * (self.sigma_max/self.sigma_min)**t


class TrainDiffussionCFG:
    def __init__(self, score_network, train_loader, val_loader, schedule, eta, lr):
        """
        Args:
        - score_network: unet.UNet instance that will be updated
        - train_loader: dataloader for the train dataset
        - val_loader: dataloader for the validation dataset
        - eta: probability of dropping label and doing unconditional matching (need for classifier-free guidance)
        - lr: learn rate for optimizer
        """
        self.eta = eta
        self.loader = train_loader
        self.val_loader = val_loader
        self.net = score_network
        self.diff_coef = schedule
        self.lr = lr
    
    def train(self, epoch_count):
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)

        for i in range(epoch_count):
            print(f'Epoch {i} ----------')
            train_losses, val_losses = [], []

            self.net.train()
            for z, y in self.loader:
                optimizer.zero_grad()
                z = z.to(device) # sample from our target distribution
                y = y.to(device=device, dtype=torch.float32) # label for that sample

                # drop labels for samples with probability self.eta
                y[torch.rand_like(y) < self.eta] = 10 # label 10 (i.e 11th label) is for unconditional

                t = torch.rand_like(y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                noise = torch.randn_like(z)

                x_t = z + self.diff_coef(t)*noise

                # for our loss function we will do L = ||score_network(x_t) - noise/self.diff_coef(t)||^2
                loss = ((self.net(x_t, t, y) - noise)**2).mean()

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # validation
            self.net.eval()
            with torch.no_grad():
                for z, y in self.val_loader:
                    z = z.to(device)
                    y = y.to(device=device, dtype=torch.float32)

                    y[torch.rand_like(y) < self.eta] = 10

                    t = torch.rand_like(y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    noise = torch.randn_like(z)

                    x_t = z + self.diff_coef(t)*noise

                    loss = ((self.net(x_t, t, y) - noise)**2).mean()

                    val_losses.append(loss.item())


            t_loss = sum(train_losses)/len(train_losses)
            v_loss = sum(val_losses)/len(val_losses)
            print(f'Train loss over {i}: {t_loss}')
            print(f'Val loss over {i}: {v_loss}')
            torch.save(self.net.state_dict(), './models/test.pth')

            with open('./models/log.csv', 'a') as f:
                f.write(f'{t_loss}, {v_loss}\n')

# We'll use Euler Maruyama method to simulate our SDE for inference
# Note that based on the training above, we get a noise_network, not a score network, so we need some adjustment
class SimulateDiff:
    def __init__(self, noise_network, schedule):
        """
        Args:
        - noise_network: trained unet.UNet instance
        """
        self.network = noise_network
        self.diff_coef = schedule
    
    @torch.no_grad
    def simulate(self, y_label, guidance_strength, timesteps):
        """
        Args:
        - y_label: int, class label for CFG
        - guidance_strength: CFG guidance strength value
        - timesteps: int
        """
        self.network.eval()

        step_size = 1/timesteps
        x = torch.randn(1, 1, 32, 32).to(device) * self.diff_coef(torch.tensor([1.0]).to(device)) # (bs, c, h, w). Also, with VE we start at N(0, sigma^2I_d) since thats where we end up at t after noising
        t = torch.ones(1, 1, 1, 1).to(device)
        y = torch.tensor(y_label).to(device)
        y_null = torch.tensor(10).to(device)
        for _ in range(timesteps):
            noise = torch.randn(1, 1, 32, 32).to(device)

            # note that we have labels 0-9 for classes, then 10 for the null (unconditional) label
            # we trained for both cond/uncond noise network using technique from CFG

            cfg_net = (1-guidance_strength)*self.network(x, t, y_null) + guidance_strength*self.network(x, t, y)

            # drift = ((self.schedule.beta(t)**2)*(self.schedule.alpha_dt(t)/self.schedule.alpha(t)) - self.schedule.beta_dt(t)*self.schedule.beta(t) + (1/2)*self.sigma(t)**2)
            # drift = drift * (-cfg_net/self.schedule.beta(t)) + x*(self.schedule.alpha_dt(t)/self.schedule.alpha(t))

            drift = -self.diff_coef(t)*cfg_net
            print(drift)

            x = x + step_size*drift + self.diff_coef(t)*math.sqrt(step_size)*noise
            t = t - step_size

        return x 


# Utility function to reverse normalization for output of inference and get back something we can render as image
def reverse_norm(x, means, stds):
    mean = torch.tensor(means).view(1, len(means), 1, 1).to(device)
    std = torch.tensor(stds).view(1, len(stds), 1, 1).to(device)
    return x * std + mean

if __name__ == '__main__':
    unet = unet.UNet([2, 4, 8], 2, 32, 10).to(device)
    # sig = DiffusionCoefficient(1)
    # sim = SimulateDiff(unet, sig)

    #print(sim.simulate(0, 100))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081]) # means and stds from data.ipynb for cifar 10 channels
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=200, shuffle=True, num_workers=2)

    validation_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    validation_loader = DataLoader(train_set, batch_size=200, shuffle=True, num_workers=2)

    train = TrainDiffussionCFG(unet, train_loader, validation_loader, DiffusionCoefficient(), 0.1, 1e-5)
    train.train(500)

    torch.save(unet.state_dict(), './models/final.pth')

    # unet.load_state_dict(torch.load('./models/test.pth', weights_only=True, map_location=torch.device(device)))

    # ns = DiffusionCoefficient()
    # sim = SimulateDiff(unet, ns)
    # y = sim.simulate(5, 1, 1000)
    # print(y)
    
    # y = reverse_norm(y, [0.1307], [0.3081])
    # y = y.clamp(0, 1).squeeze(0) # c, h, w

    # y = y.permute(1, 2, 0).cpu() # convert to h, w, c 

    # plt.imshow(y)
    # plt.show()