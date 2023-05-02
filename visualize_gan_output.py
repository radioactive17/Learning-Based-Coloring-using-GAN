

import os
import glob
import time
import numpy as np
import pickle
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnetUnit(nn.Module):
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

class Discriminator(nn.Module):
  def __init__(self, c, nf=64, nd=3):
        super().__init__()

        model = [self.ret_layers(c,nf,norm_=False)]
        model = model + [self.ret_layers(nf*2**i,nf*2**(i+1),s=1 if i==(nd-1) else 2) for i in range(nd)]
        model = model + [self.ret_layers(nf*2**nd,1,s=1,norm_=False,act_=False)]

        self.model = nn.Sequential(*model)


  def ret_layers(self,n1,n2,k=4,s=2,p=1,norm_=True,act_=True):
    layers = [nn.Conv2d(n1,n2,k,s,p,bias=not norm_)]
    if norm_:
      layers += [nn.BatchNorm2d(n2)]

    if act_: 
      layers  +=[nn.LeakyReLU(0.2,True)]

    return nn.Sequential(*layers)

  
  def forward(self,x):
    return self.model(x)

class LossForGAN(nn.Module):
  def __init__(self, real=1.0, synth=0.0):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.real = torch.tensor(real).to(device)
        self.synth = torch.tensor(synth).to(device)
        
  
  def __call__(self,preds,is_target_real):
    if is_target_real:
      labels = self.real
    else:
      labels = self.synth

    labels = labels.expand_as(preds)
    loss = self.loss(preds,labels)
    return loss

def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

class GANModel(nn.Module):
  def __init__(self, G=None, D=None, lr=2e-4,
                 b1=0.5, b2=0.999, lamb=100.):
        super().__init__()

        self.loss_map = {'dis_synth':0,'dis_real':0,'dis':0,'bce':0,'L1':0,'gen':0}

        if G is not None:
          self.G = G.to(device)
        
        else:
          self.G = init_model(Generator(ic=1, oc=2, nd=8, nf=64),device)

        self.D = init_model(Discriminator(c=3, nd=3, nf=64),device)
        self.GANcriterion = LossForGAN().to(device)
        self.L1 = nn.L1Loss().to(device)
        self.optG = optim.Adam(self.G.parameters(),lr = lr, betas=(b1,b2))
        self.optD = optim.Adam(self.D.parameters(),lr = lr, betas=(b1,b2))
        self.lamb = lamb

  def req_grad(self,m,req=True):
    for param in m.parameters():
      param.requires_grad = req
  
  def get_L_and_ab(self,data):
    self.L = data['L'].to(device)
    self.ab = data['ab'].to(device)
  
  def forward(self):
    self.synth_color = self.G(self.L)

  def backprop_dis(self):
    synth_img = torch.cat([self.L,self.synth_color],dim=1)
    synth_pred = self.D(synth_img.detach())

    self.dis_synth_loss = self.GANcriterion(synth_pred,False)


    real_img = torch.cat([self.L,self.ab],dim=1)
    real_pred = self.D(synth_img.detach())

    self.dis_real_loss = self.GANcriterion(synth_pred,True)
    self.dis_loss = (self.dis_synth_loss + self.dis_real_loss)*0.5
    self.dis_loss.backward()

  def backprop_gen(self):
    synth_img = torch.cat([self.L,self.synth_color],dim=1)
    synth_pred = self.D(synth_img)

    self.loss_bce = self.GANcriterion(synth_pred,True)
    self.loss_L1 = self.L1(self.synth_color,self.ab)*self.lamb
    self.gen_Loss = self.loss_bce + self.loss_L1
    self.gen_Loss.backward()

  def train_dis(self):
    self.D.train()
    self.req_grad(self.D,True)
    self.optD.zero_grad()
    self.backprop_dis()
    self.optD.step()

  def train_gen(self):
    self.G.train()
    self.req_grad(self.D,False)
    self.optG.zero_grad()
    self.backprop_gen()
    self.optG.step()

  def update_losses(self):
    self.loss_map['dis_synth']+=self.dis_synth_loss.item()
    self.loss_map['dis_real']+=self.dis_real_loss.item()
    self.loss_map['dis'] += self.dis_loss.item()
    self.loss_map['bce'] += self.loss_bce.item()
    self.loss_map['L1'] += self.loss_L1.item()
    self.loss_map['gen']+= self.gen_Loss.item()

  def reset_loss(self):
    self.loss_map = {'dis_synth':0,'dis_real':0,'dis':0,'bce':0,'L1':0,'gen':0}


  def optimize(self):
    self.forward()
    self.train_dis()
    self.train_gen()

def lab_to_rgb(L, ab):
    L = (L + 1.)*50.
    ab = ab*110.
    L_ab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in L_ab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def visualize(m,data):
    m.G.eval()
    with torch.no_grad():
        m.get_L_and_ab(data)
        m.forward()
    m.G.train()
    synth_color = m.synth_color.detach()
    real_color = m.ab
    L = m.L
    synth_imgs = lab_to_rgb(L,synth_color)
    real_imgs = lab_to_rgb(L,real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(synth_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    fig.savefig(f"results/colorization{time.time()}.png")


with open('data/raw_test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    G = DynamicUnet(body, n_output, (size, size)).to(device)
    return G

G = build_res_unet(n_input=1, n_output=2, size=256)
checkpoint = torch.load("models/generator_dunet_checkpoint.pt",map_location=device)
G.load_state_dict(checkpoint['model_state_dict'])

m = GANModel(G=G)
for d in tqdm(test_data):
    visualize(m,d)


