# Discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F

class D_WGAN(nn.Module):
  def __init__(self):
    super(D_WGAN, self).__init__()
    self.leaky=nn.LeakyReLU(0.2, inplace=True)
    self.x_conv1=nn.Conv2d(1,50,4,2,1) # (1,1,384,384)->(1,50,192,192)
    self.x_bn1=nn.BatchNorm2d(50)
    self.x_conv2=nn.Conv2d(50,50,4,2,1) # (1,50,192,192)->(1,50,96,96)
    self.x_bn2=nn.BatchNorm2d(50)
    self.x_conv3=nn.Conv2d(50,50,4,2,1) # (1,50,96,96)->(1,50,48,48)
    self.x_bn3=nn.BatchNorm2d(50)
    self.x_conv4=nn.Conv2d(50,50,4,2,1) # (1,50,48,48)->(1,50,24,24)
    self.x_bn4=nn.BatchNorm2d(50)
    self.x_conv5=nn.Conv2d(50,50,4,2,1) # (1,50,24,24)->(1,50,12,12)
    self.x_bn5=nn.BatchNorm2d(50)
    self.x_conv6=nn.Conv2d(50,25,4,2,1) # (1,25,12,12)->(1,25,6,6)
    self.x_bn6=nn.BatchNorm2d(25)
    
    self.y_conv1=nn.Conv2d(1,50,4,2,1) # (1,1,384,384)->(1,50,192,192)
    self.y_bn1=nn.BatchNorm2d(50)
    self.y_conv2=nn.Conv2d(50,50,4,2,1) # (1,50,192,192)->(1,50,96,96)
    self.y_bn2=nn.BatchNorm2d(50)
    self.y_conv3=nn.Conv2d(50,50,4,2,1) # (1,50,96,96)->(1,50,48,48)
    self.y_bn3=nn.BatchNorm2d(50)
    self.y_conv4=nn.Conv2d(50,50,4,2,1) # (1,50,48,48)->(1,50,24,24)
    self.y_bn4=nn.BatchNorm2d(50)
    self.y_conv5=nn.Conv2d(50,50,4,2,1) # (1,50,24,24)->(1,50,12,12)
    self.y_bn5=nn.BatchNorm2d(50)
    self.y_conv6=nn.Conv2d(50,25,4,2,1) # (1,50,12,12)->(1,50,6,6)
    self.y_bn6=nn.BatchNorm2d(25)
    
    self.fc1=nn.Linear(2*25*6*6,1000)
    self.fc2=nn.Linear(1000,1)
    #self.bn7=nn.BatchNorm1d(1000)

  def forward(self,x,y):
    x=self.x_conv1(x)
    x=self.x_bn1(x)
    x=self.leaky(x)
    x=self.x_conv2(x)
    x=self.x_bn2(x)
    x=self.leaky(x)
    x=self.x_conv3(x)
    x=self.x_bn3(x)
    x=self.leaky(x)
    x=self.x_conv4(x)
    x=self.x_bn4(x)
    x=self.leaky(x)
    x=self.x_conv5(x)
    x=self.x_bn5(x)
    x=self.leaky(x)
    x=self.x_conv6(x)
    x=self.x_bn6(x)

    # heatmap : batch*num_heatmaps*input_w*input_h
    y=self.y_conv1(y)
    y=self.y_bn1(y)
    y=self.leaky(y)
    y=self.y_conv2(y)
    y=self.y_bn2(y)
    y=self.leaky(y)
    y=self.y_conv3(y)
    y=self.y_bn3(y)
    y=self.leaky(y)
    y=self.y_conv4(y)
    y=self.y_bn4(y)
    y=self.leaky(y)
    y=self.y_conv5(y)
    y=self.y_bn5(y)
    y=self.leaky(y)
    y=self.y_conv6(y)
    y=self.y_bn6(y)
        
    x=x.view(1,-1)
    y=y.view(1,-1)
    x=torch.cat([x,y],dim=1)

    x=self.fc1(x)
    #x=self.bn7(x)
    x=F.relu(x)
    x=self.fc2(x)

    x=x.view(-1)
    return x