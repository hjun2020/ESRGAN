"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import RRDBNet_arch as arch
import torch.quantization

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--saved_model_path", required=False, type=str, default='', help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')


pretrained_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# if not opt.saved_model_path:
pretrained_model.load_state_dict(torch.load(model_path), strict=True)
# pretrained_model = pretrained_model.to(device)


# Define the new layers to add to the model
new_layers = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
)

# Define the complete model
model = nn.Sequential(
    pretrained_model,
    new_layers
)
model.load_state_dict(torch.load(opt.saved_model_path), strict=True)


# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model,  # the loaded PyTorch model
    {torch.nn.Conv2d},  # specify which layers to quantize
    dtype=torch.quint8  # specify the datatype for quantization
)

quantized_model.to(device)
# Save the quantized model
torch.jit.save(torch.jit.script(quantized_model), 'quantized_model.pt')

