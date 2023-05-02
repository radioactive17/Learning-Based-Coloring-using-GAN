

import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_L(img_path):
  T = transforms.Resize((256, 256),Image.BICUBIC)
  img = Image.open(img_path).convert("RGB")
  img = T(img)
  img = np.array(img)
  img_lab = rgb2lab(img).astype("float32")
  img_lab = transforms.ToTensor()(img_lab)
  L = img_lab[[0], ...] / 50. - 1.
  ab = img_lab[[1, 2], ...] / 110.

  return L

L = get_L('test4.jpeg')

class UnetUnit(nn.Module):
  # nf:n1
  # ni:n2
  def __init__(self, n1, n2, sub=None, c=None, dropout=False,
                 inner=False, outer=False):
    super().__init__()

    if c is None:
      c = n1

    dc = nn.Conv2d(c,n2,kernel_size=4,stride=2,padding=1,bias=False)
    dr = nn.LeakyReLU(0.2,True)
    dn = nn.BatchNorm2d(n2)

    ur = nn.ReLU(True)
    un = nn.BatchNorm2d(n1)

    if outer:
      uc = nn.ConvTranspose2d(2*n2,n1,kernel_size=4,stride=2,padding=1)
      model = [dc,sub,ur,uc,nn.Tanh()]

    elif inner:
      uc = nn.ConvTranspose2d(n2,n1,kernel_size=4,stride=2,padding=1,bias=False)
      model = [dr,dc,ur,uc,un]

    else:
      uc = nn.ConvTranspose2d(2*n2,n1,kernel_size=4,stride=2,padding=1,bias=False)
      temp = [dr,dc,dn,sub,ur,uc,un]

      if dropout:
        model = temp + [nn.Dropout(0.5)]
      
      else:
        model = temp
    self.outer = outer
    self.model = nn.Sequential(*model)

  def forward(self,x):
    if self.outer:
      return self.model(x)
    
    else:
      return torch.cat([x,self.model(x)],1)

class Generator(nn.Module):
  def __init__(self, ic=1, oc=2, nd=8, nf=64):
        super().__init__()

        uu = UnetUnit(8*nf,8*nf,inner=True)
        for i in range(3):
          uu = UnetUnit(8*nf,8*nf,sub=uu,dropout=True)
        of = 8*nf
        for i in range(3):
          uu = UnetUnit(of//2,of,sub=uu)
          of = of//2
        self.model = UnetUnit(oc,of,c=ic,sub=uu,outer=True)

  def forward(self,x):
    return self.model(x)

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

def build_res_unet(n_input=1,n_output=2,size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    G = DynamicUnet(body, n_output, (size, size)).to(device)
    return G

G = build_res_unet(n_input=1,n_output=2,size=256)
checkpoint = torch.load("models/generator_dunet_checkpoint.pt",map_location = device)
G.load_state_dict(checkpoint['model_state_dict'])


L = L.unsqueeze(0)
with torch.no_grad():
  ab = G.forward(L)
ab = ab.detach()
predicted_img = lab_to_rgb(L,ab)
fig = plt.figure()
plt.imshow(predicted_img[0])
