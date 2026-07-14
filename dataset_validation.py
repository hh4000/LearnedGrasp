import torch
import sys, os
from pathlib import Path
from grasp_image_set import GraspImagePosesDataset
from tqdm import tqdm

data_path = Path("~/Documents/P10/hot3d/data/train_images_poses.pt").expanduser()
data = torch.load(data_path, weights_only=False)
input_means = torch.zeros(4)
input_vars = torch.zeros(4)
output_means = torch.zeros(15)
output_vars = torch.zeros(15)
n_samples = len(data)
for i in tqdm(range(n_samples), desc="Calculating means..."):
    img, lbl = data[i]
    img: torch.Tensor
    lbl: torch.Tensor
    input_means+=img.mean(dim=(1,2))/n_samples
    input_vars+= img.pow(2).mean(dim=(1, 2)) / n_samples
    output_means+=lbl/n_samples
    output_vars+= lbl.pow(2) / n_samples
input_vars -= input_means.pow(2)
output_vars -= output_means.pow(2)
print(input_means, output_means)
print(input_vars, output_vars)
