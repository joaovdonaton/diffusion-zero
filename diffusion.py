# If ran as main serves as our training script.

import torch
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import unet
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torch.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NoiseScheduler:
    def __init__(self):
        self.beta_min = 0.1
        self.beta_max = 20

    def beta(self, t):
        return self.beta_min + t*(self.beta_max - self.beta_min)

    # a_bar = e^(-int_0^t(beta)dt)
    def alpha_bar(self, t):
        return torch.exp(-(self.beta_min*t + 0.5*(self.beta_max-self.beta_min) * (t**2)))


class TrainDiffussionCFG:
    def __init__(self, score_network, train_loader, val_loader, schedule, eta, lr, out_path):
        """
        Args:
        - score_network: unet.UNet instance that will be updated
        - train_loader: dataloader for the train dataset
        - val_loader: dataloader for the validation dataset
        - eta: probability of dropping label and doing unconditional matching (need for classifier-free guidance)
        - schedule: noise scheduler for loss
        - lr: learn rate for optimizer
        - out_path: path to write logs and save models
        """
        self.eta = eta
        self.loader = train_loader
        self.val_loader = val_loader
        self.net = score_network
        self.schedule = schedule
        self.lr = lr
        self.out_path = out_path
    
    def train(self, epoch_count):
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)
        scaler = GradScaler('cuda')

        val_loss_history = []
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

                alpha_bar = self.schedule.alpha_bar(t).clamp(min=1e-5, max=0.999)

                x_t = torch.sqrt(alpha_bar)*z + torch.sqrt(1-alpha_bar)*noise

                # DEBUG VISUALIZE CORRUPTION
                # x_tt = reverse_norm(x_t[0], [0.1307], [0.3081])
                # x_tt = x_tt.clamp(0, 1).squeeze(0) # c, h, w

                # x_tt = x_tt.permute(1, 2, 0).cpu() # convert to h, w, c 
                # print(x_tt)
                # print(f'at t = {t[0]}')

                # plt.imshow(x_tt, cmap='gray')
                # plt.show()
                ###

                with autocast('cuda'):
                    net_eval = self.net(x_t, t, y)
                    # this setup causes us to learn the negative score
                    loss = ( (torch.sqrt(1-alpha_bar)*net_eval + (noise))**2).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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

                    alpha_bar = self.schedule.alpha_bar(t).clamp(min=1e-5, max=0.999)

                    x_t = torch.sqrt(alpha_bar)*z + torch.sqrt(1-alpha_bar)*noise

                    loss = ( (torch.sqrt(1-alpha_bar)*self.net(x_t, t, y) + (noise))**2).mean()

                    val_losses.append(loss.item())


            t_loss = sum(train_losses)/len(train_losses)
            v_loss = sum(val_losses)/len(val_losses)
            val_loss_history.append(v_loss)
            print(f'Train loss over {i}: {t_loss}')
            print(f'Val loss over {i}: {v_loss}')

            if len(val_loss_history) == 1 or (val_loss_history[-1] < max(val_loss_history[:-1])):
                print('New best val loss, writing model...')
                torch.save(self.net.state_dict(), f'{self.out_path}bestval-{i}.pth')

            with open(f'{self.out_path}log.csv', 'a') as f:
                f.write(f'{t_loss}, {v_loss}\n')

# We'll use Euler Maruyama method to simulate our SDE for inference
class SimulateDiff:
    def __init__(self, score_network, schedule):
        """
        Args:
        - noise_network: trained unet.UNet instance
        """
        self.network = score_network
        self.schedule = schedule
    
    @torch.no_grad
    def simulate(self, y_label, guidance_strength, timesteps):
        """
        Args:
        - y_label: int, class label for CFG
        - guidance_strength: CFG guidance strength value
        - timesteps: int
        """
        self.network.eval()

        step_size = 1.0/timesteps
        x = torch.randn(1, 3, 32, 32).to(device)
        t = torch.full((1, 1, 1, 1), 0.999).to(device)
        y = torch.tensor(y_label).to(device)
        y_null = torch.tensor(10).to(device)
        for debug_t in range(timesteps):
            noise = torch.randn(1, 3, 32, 32).to(device)

            # note that we have labels 0-9 for classes, then 10 for the null (unconditional) label

            cfg_net = (1-guidance_strength)*self.network(x, t, y_null) + guidance_strength*self.network(x, t, y)

            bt = self.schedule.beta(t)
            alpha_bar = self.schedule.alpha_bar(t).clamp(min=1e-5, max=0.999)

            drift = -0.5 * bt * x + bt*cfg_net

            x = x + step_size*drift + torch.sqrt(step_size*bt)*noise
            t = t - step_size

            # if debug_t % 90011 == 0 and debug_t != 0:
            #     # DEBUG VISUALIZE denoising
            #     x_tt = reverse_norm(x[0], [0.1307], [0.3081])
            #     x_tt = x_tt.clamp(0, 1).squeeze(0) # c, h, w

            #     x_tt = x_tt.permute(1, 2, 0).cpu() # convert to h, w, c 
            #     print(x_tt)
            #     print(f'at t = {t[0]}')

            #     plt.imshow(x_tt, cmap='gray')
            #     plt.show()

        return x 


# Utility function to reverse normalization for output of inference and get back something we can render as image
def reverse_norm(x, means, stds):
    mean = torch.tensor(means).view(1, len(means), 1, 1).to(device)
    std = torch.tensor(stds).view(1, len(stds), 1, 1).to(device)
    return x * std + mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--learn_rate', default=1e-4, required=False)
    parser.add_argument('-b', '--batch_size', default=400, required=False)
    parser.add_argument('-e', '--epochs', default=500, required=False)
    parser.add_argument('-d', '--drop_rate', default=0.1, help='eta probability of dropping label for our classifier-free guidance set up', required=False)
    parser.add_argument('-o', '--out_path', help='path for directory to output log and model training checkpoints', required=True)

    args = parser.parse_args()

    LEARN_RATE = float(args.learn_rate)
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
    ETA = float(args.drop_rate)
    OUT_PATH = args.out_path if args.out_path.endswith('/') else args.out_path+'/'

    if ETA < 0 or ETA > 1:
        print('Eta (-d, --drop_rate) value be in (0,1)')
        exit()

    unet = unet.UNet([64, 128, 256], 2, 128, 32).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]) # means and stds from data.ipynb 
    ])

    # train_set = (datasets.MNIST(root='./data', train=True, download=True, transform=transform))
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # validation_set = (datasets.MNIST(root='./data', train=False, download=True, transform=transform))
    # validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    train_set = (datasets.CIFAR10(root='./data', train=True, download=True, transform=transform))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    validation_set = (datasets.CIFAR10(root='./data', train=False, download=True, transform=transform))
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    ns = NoiseScheduler()
    train = TrainDiffussionCFG(unet, train_loader, validation_loader, ns, ETA, LEARN_RATE, OUT_PATH)

    # save train details
    with open(f'{OUT_PATH}details.txt', 'w') as f:
        f.write(f'Train sample count: {len(train_loader)}\n')
        f.write(f'Validation sample count: {len(validation_loader)}\n')
        f.write(f'Learn Rate: {LEARN_RATE}\n')
        f.write(f'Batch size: {BATCH_SIZE}\n')
        f.write(f'Eta rate: {ETA}\n')
        f.write(f'Dataset: MNIST Digits\n')
        f.write(f'Beta min: {ns.beta_min}\n')
        f.write(f'Beta max: {ns.beta_max}\n')

    train.train(EPOCHS)

    torch.save(unet.state_dict(), f'{OUT_PATH}final.pth')