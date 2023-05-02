
import os
import glob
import time
import numpy as np
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

# {'dis_synth':0,'dis_real':0,:'dis':0,'bce':0,'L1':0,'gen':0}
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

def average_out_loss(dict_,iters_):
  for k in dict_.keys():
    dict_[k] = dict_[k]/iters_
  return dict_

def trainer(m,train_data,epochs=100):
  for e in range(epochs):
    for data in tqdm(train_data):
      m.get_L_and_ab(data)
      m.optimize()
      m.update_losses()
    metrics.append(average_out_loss(m.loss_map,1000))
    m.reset_loss()
    
class L2ABDataset(Dataset):
    def __init__(self, path, split='train'):
        if split == 'train':
            self.T = transforms.Compose([transforms.Resize((256, 256),  Image.BICUBIC),transforms.RandomHorizontalFlip()])
        else:
            self.T = transforms.Resize((256, 256),  Image.BICUBIC)

        self.path = path
    
    def __getitem__(self, i):
        img = Image.open(self.path[i]).convert("RGB")
        img = self.T(img)
        img = np.array(img)
        lab_img = rgb2lab(img).astype("float32")
        lab_img = transforms.ToTensor()(lab_img)
        L = lab_img[[0], ...] / 50. - 1.
        ab = lab_img[[1, 2], ...] / 110.

        return {'L': L, 'ab': ab}  

    def __len__(self):
        return len(self.path)

from fastai.data.external import untar_data, URLs
coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"

paths = glob.glob(coco_path+"/*.jpg")
np.random.seed(456)
data_paths = np.random.choice(paths,20000,replace=False)
ri = np.random.permutation(20_000)
train_i = ri[:16000]
test_i = ri[16000:]
train_paths = data_paths[train_i]
val_paths = data_paths[test_i]

t1 = L2ABDataset(train_paths,split="train")
train_imgs = DataLoader(t1,batch_size=16)
t2 = L2ABDataset(val_paths,split="val")
val_imgs = DataLoader(t2,batch_size=16)

m = GANModel()
trainer(m,train_imgs,epochs=100)

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    G = DynamicUnet(body, n_output, (size, size)).to(device)
    return G

def pretrain_gen(G,train_imgs,epochs):
  optimizer = optim.Adam(G.parameteres(),lr=1e-4)
  loss_fn = nn.L1Loss()
  for e in range(epochs):
    total_loss = 0
    for data in tqdm(train_imgs):
      L = data['L'].to(device)
      ab = data['ab'].to(device)

      syn_ab = G(L)
      loss = loss_fn(syn_ab,ab)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    

G = build_res_unet(n_input=1, n_output=2, size=256)
pretrain_generator(G,train_imgs,30)
torch.save(G.state_dict(), "models/res18-unet.pt")

G = build_res_unet(n_input=1, n_output=2, size=256)
G.load_state_dict(torch.load("models/res18-unet.pt"))
m = GANModel(G=G)
trainer(m,epochs=30,train_imgs) # It converges in 30 epochs
