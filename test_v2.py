import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import argparse
import torch.nn as nn

parser = argparse.ArgumentParser()



parser.add_argument("--test_img_folder", type=str, default="LR", help="name of the dataset")
opt = parser.parse_args()
print(opt)


model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = f'{opt.test_img_folder}/*'
print(test_img_folder)


# Define the pre-trained model
pretrained_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# pretrained_model.load_state_dict(torch.load(model_path), strict=True)
# pretrained_model.eval()

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
model.load_state_dict(torch.load('saved_models/generator_45.pth'), strict=True)
model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
