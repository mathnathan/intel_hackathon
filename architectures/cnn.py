import input_data as g  # global variables
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
#from torchvision import datasets, transforms
#from torchvision.utils import save_image
#from sklearn.neighbors import KDTree
import os, sys
import numpy as np
#import utils as u
#from IPython import embed

# Read in images, output images.
class CNN(nn.Module):
    def __init__(self, params):
        super(CNN, self).__init__()

        # single datum (figure, histogram, etc.)
        self.datum_shape = params["datum_shape"]
        print("cnn: self.datum_shape ", self.datum_shape)

        self.CNN  = g.CNN
        self.LATENT_DIM = params['LATENT_DIM']
        self.BATCH_SIZE = params['BATCH_SIZE']

# formula:
# Input: (N,Cin,Lin)
# Output: (N,Cout,Lout)
# where Lout=floor((Lin+2*padding-dilation*(kernel_size-1)-1)/stride+1)

# stride=2, dilation=1, kernel_size=3, padding=1
# Lout = floor( (Lin+2-1*(3-1)-1)/2 + 1) = floor((Lin-1)/2 + 1) ==> 127+1 = 128
# stride=2, dilation=1, kernel_size=5, padding=1
# Lout = floor( (Lin+2-1*(5-1)-1)/2 + 1) = floor((Lin-3)/2 + 1) ==> 12w+1 = 127


        self.nb_conv = len(params['out_channels']) - 1
        self.ch = params['out_channels']
        self.st = params['strides']
        self.di = params['dilation']
        self.ks = params['kernel_size']
        self.pa = params['padding']
        self.fc = params['fc_out_chan']

        if len(self.st) == 1: self.st *= self.nb_conv
        if len(self.di) == 1: self.di *= self.nb_conv
        if len(self.ks) == 1: self.ks *= self.nb_conv
        if len(self.pa) == 1: self.pa *= self.nb_conv

        #ks = 3;           self.ks = kernel_size  = [(ks,ks)] * self.nb_conv
        #pa = (ks-1) // 2; self.pa = padding      = [(pa,pa)] * self.nb_conv

        # necessary to allow for self.ch[i+1] in call to Conv2d
        assert(len(self.ch) > self.nb_conv)

        self.conv    = {}
        self.deconv  = {}
        self.bn      = {}
        self.dbn     = {}
        self.indices = {}

        print(self.datum_shape)
        self.w = self.datum_shape[3]

        for i in range(self.nb_conv):
            self.conv[i] = nn.Conv2d(self.ch[i], self.ch[i+1], kernel_size=self.ks[i], padding=self.pa[i], stride=self.st[i])
            self.w = self.w // 2
            nn.init.xavier_uniform_(self.conv[i].weight)
            self.bn[i] = nn.BatchNorm2d(self.ch[i+1])
            print(self.conv[i])

        # full layer parameters:
        # out_features: [c1, c2, ...] (last one is LATENT_DIM or whatever)
        self.cnn_in_features = self.w**2 * self.ch[self.nb_conv]

        layers = []
        nb_fc = len(self.fc)

        if len(self.fc) > 0:
            layers.append(nn.Linear(self.cnn_in_features, self.fc[1]))
        for i in range(1,nb_fc-1):
            layers.append(nn.Linear(self.fc[i], self.fc[i+1]))
        self.fc_layers = nn.Sequential(*layers)

        #self.fc21 = nn.Linear(self.cnn_in_features, self.LATENT_DIM)
        #self.fc22 = nn.Linear(self.cnn_in_features, self.LATENT_DIM)

    #------------------------
    def forward(self, x):
        # conv expects x[batch, nb_channels, w, h]
        #c, w, h = x.shape[1:] # not required for now

        for i in range(self.nb_conv):
            x = self.conv[i](x)
            c, w, h = x.shape[1:]
            x = F.relu(x)
            print("x.shape: ", x.shape)

        x = x.view(x.shape[0], -1)
        print("cnn, x.shape: ", x.shape)

        x = self.fc_layers(x)
        print("cnn, after fc, x.shape: ", x.shape)

        self.out_shape = x.shape
        print("cnn, x.shape: ", x.shape)
        return x

#----------------------------------------------------------------------
