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

parser = argparse.ArgumentParser()



parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
opt = parser.parse_args()
print(opt)




Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
hr_shape = (opt.hr_height, opt.hr_height)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = 'models/RRDB_ESRGAN_x4.pth'
# Define the pre-trained model
pretrained_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
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

criterion_pixel = torch.nn.L1Loss().to(device)

# Freeze the pre-trained model weights
for param in pretrained_model.parameters():
    param.requires_grad = False

# Define the optimizer to only update the new layers
optimizer = optim.Adam(new_layers.parameters(), lr=0.001)

dataloader = DataLoader(
    ImageDataset("/home/ubuntu/PyTorch-GAN/data/BSR/BSDS500/data/images/train", hr_shape=hr_shape),
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

        inputs = Variable(imgs["lr"].type(Tensor))
        labels = Variable(imgs["hr"].type(Tensor))

        optimizer.zero_grad()

        outputs = model(inputs)

        # Compute the loss only on the new layers
        loss = criterion_pixel(outputs, labels)

        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
