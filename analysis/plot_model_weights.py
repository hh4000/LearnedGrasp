import numpy as np
import torch
import matplotlib.pyplot as plt

data =torch.load('models/260504_134312.pth')
for k,v in data.items():
    print(k,v.shape)
fig, axes = plt.subplots(3,4)
vmin = np.inf
vmax = -np.inf

for i in range(3):
    for j in range(2):
        j*=2
        vmin = min(vmin, data[f"gnns.{i}.nn.{j}.weight"].cpu().min())
        vmax = max(vmin, data[f"gnns.{i}.nn.{j}.weight"].cpu().max())

for i in range(3):
    for j in range(2):
        j*=2
        im = axes[i][j].imshow(data[f'gnns.{i}.nn.{j}.weight'].cpu(), vmin=vmin, vmax = vmax)

        axes[i][j+1].bar(range(64), data[f'gnns.{i}.nn.{j}.bias'].cpu())
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im,cax=cbar_ax)
plt.show()


exit()
