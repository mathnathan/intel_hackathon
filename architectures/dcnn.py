import input_data as g  # global variables
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import os, sys
import numpy as np


class DCNN(nn.Module):
    def __init__(self, params):
        super(DCNN, self).__init__()

        # single datum (figure, histogram, etc.)
        self.datum_shape = params["datum_shape"]

        self.CNN  = g.CNN
        self.LATENT_DIM = params['LATENT_DIM']
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.w = params['w']

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
        self.op = params['output_padding']
        self.fc = params['fc_out_chan']

        if len(self.st) == 1: self.st *= self.nb_conv
        if len(self.di) == 1: self.di *= self.nb_conv
        if len(self.ks) == 1: self.ks *= self.nb_conv
        if len(self.pa) == 1: self.pa *= self.nb_conv
        if len(self.op) == 1: self.op *= self.nb_conv

        # necessary to allow for self.ch[i+1] in call to Conv2d
        assert(len(self.ch) > self.nb_conv)

        self.conv = {}
        self.deconv = {}
        self.bn = {}
        self.dbn = {}
        self.indices = {}
        self.size = {}

        self.dcnn_out_features = params['out_features']  # NOT ELEGANT
        self.dfc1 = nn.Linear(self.LATENT_DIM, self.dcnn_out_features)

        for i in range(self.nb_conv):
            self.deconv[i] = nn.ConvTranspose2d(self.ch[i], self.ch[i+1], kernel_size=self.ks[i], padding=self.pa[i], dilation=self.di[i], stride=self.st[i], output_padding=self.op[i])
            nn.init.xavier_uniform_(self.deconv[i].weight)
            self.dbn[i] = nn.BatchNorm2d(self.ch[i])

        layers = []
        nb_fc = len(self.fc)

        print("=====================")
        print("self.fc= ", self.fc) # 10, 32, -1
        for i in range(0,nb_fc-1): # (10, 32),    (32, -1)
            if self.fc[i+1] == -1:
                layers.append(nn.Linear(self.fc[i], self.dcnn_out_features))
                print("if i= ", self.fc[i], self.dcnn_out_features)
            else:
                print("else -- ", i, self.fc[i], self.fc[i+1])     # -1
                layers.append(nn.Linear(self.fc[i], self.fc[i+1]))
        #sys.exit()

        self.fc_layers = nn.Sequential(*layers)

    #----------------------
    def forward(self, x):
        print("dcnn, x: ", x.shape)
        x = self.fc_layers(x)
        print("dcnn, after fc, x: ", x.shape)

        #  self.w is fine if there are no fully-connected layers. What to do if there are?
        x = x.view(x.shape[0], -1, self.w, self.w)    # REMOVE HARDCODING (for 64x64 images)
        print("dcnn,after viewx: ", x.shape)

        #print("dcnn,nb_conv: ", self.nb_conv)
        for i in range(self.nb_conv):
            x = self.deconv[i](x)
            #print("dcnn,after deconv: ", self.deconv[i])
            print("dcnn,after deconv: ", x.shape)
            if i > 0:
                x = F.relu(x)

        x = F.sigmoid(x)
        self.out_shape = x.shape # out [batch,:]
        print("dcnn, x: ", x.shape)

        return x

#----------------------------------------------------------------------

