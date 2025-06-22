import torch
import unet
from diffusion import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-w', '--guidance_strength', help='guidance strength for CFG', required=True)
parser.add_argument('-t', '--timesteps', help='timesteps for euler maruyama method', required=True)

args = parser.parse_args()

GUIDANCE_STRENGTH = float(args.guidance_strength)
TIMESTEPS = int(args.timesteps)

MODEL_PATH = args.model

unet = unet.UNet([64, 128, 256], 2, 128, 32).to(device)

# unet.load_state_dict(torch.load('./models/test110.pth', weights_only=True, map_location=torch.device(device)))

# ns = NoiseScheduler()
# sim = SimulateDiff(unet, ns)
# y = sim.simulate(9, 3, 200)
# print(y)

# y = reverse_norm(y, [0.1307], [0.3081])
# y = y.clamp(0, 1).squeeze(0) # c, h, w

# y = y.permute(1, 2, 0).cpu() # convert to h, w, c 
# print(y)

# plt.imshow(y, cmap='gray')
# plt.show()

unet.load_state_dict(torch.load(f'{MODEL_PATH}', weights_only=True, map_location=torch.device(device)))

ns = NoiseScheduler()
sim = SimulateDiff(unet, ns)

outs = []

print('Generating 10 samples')
for i in tqdm(range(10)):
    y = sim.simulate(i, GUIDANCE_STRENGTH, TIMESTEPS)
    
    y = reverse_norm(y, [0.1307], [0.3081])
    y = y.clamp(0, 1).squeeze(0) # c, h, w

    y = y.permute(1, 2, 0).cpu() # convert to h, w, c 

    outs.append(y)

fig, axes = plt.subplots(1, 10, figsize=(20, 2))

for ax, o in zip(axes, outs):
    ax.imshow(o, cmap='gray', interpolation='nearest')
    ax.axis('off')   # turn off ticks

plt.tight_layout()
plt.show()