import torch.nn as nn
import torch.optim as optim
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from torch.utils.data import DataLoader
import argparse
from datasets import *
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

parser = argparse.ArgumentParser()



parser.add_argument("--dataset_folder", type=str, default="/home/ubuntu/PyTorch-GAN/data/BSR/BSDS500/data/images/train", help="name of the dataset")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--saved_model_path", type=str, default="", help="name of the dataset")
opt = parser.parse_args()
print(opt)




Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
hr_shape = (opt.hr_height, opt.hr_height)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = 'models/RRDB_ESRGAN_x4.pth'
# Define the pre-trained model
pretrained_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
if not opt.saved_model_path:
    pretrained_model.load_state_dict(torch.load(model_path), strict=True)
pretrained_model.eval()

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
model.to(device)

if opt.saved_model_path:
    model.load_state_dict(torch.load(opt.saved_model_path), strict=True)

criterion_pixel = torch.nn.L1Loss().to(device)

# Freeze the pre-trained model weights
for param in pretrained_model.parameters():
    param.requires_grad = False

# Define the optimizer to only update the new layers
optimizer = optim.Adam(new_layers.parameters(), lr=0.001)

dataloader = DataLoader(
    ImageDataset(opt.dataset_folder, hr_shape=hr_shape),
    batch_size=4,
    shuffle=True,
    num_workers=4,
)
# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    for i, imgs in enumerate(dataloader):
        # inputs, labels = data
        # inputs = inputs.to(device)
        # labels = labels.to(device)
        batches_done = epoch * len(dataloader) + i

        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        optimizer.zero_grad()

        gen_hr = model(imgs_lr)

        # Compute the loss only on the new layers
        loss = criterion_pixel(gen_hr, imgs_hr)

        loss.backward()
        optimizer.step()

        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            # img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)

        if batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(model.state_dict(), "saved_models/generator_%d.pth" % epoch)

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
