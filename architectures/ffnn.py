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
import utils as u
#from IPython import embed

# Feed-forward networks
# should work with any kind of data
class FFNN(nn.Module):
    def __init__(self, params):
        super(FFNN, self).__init__()

        # single datum (figure, histogram, etc.)
        self.datum_shape = params["datum_shape"]
        self.datum_out_shape = params["datum_out_shape"]
        print("ffnn: self.datum_shape ", self.datum_shape)

# formula:
# Input: (N,Cin,Lin)
# Output: (N,Cout,Lout)


        # list of in and out feature (nb ofnodesin layer)
        self.fc = params['fc_out_chan']

        #print(self.datum_shape)

        layers = []

        print("++++")
        nb_fc = len(self.fc)
        print("nb_fc= ", nb_fc)
        for i, f in enumerate(self.fc):
            #print("features: ", f[0], f[1])
            linear = nn.Linear(f[0], f[1])
            print("add linear")
            nn.init.xavier_uniform_(linear.weight)
            layers.append(linear)
            if i < (nb_fc-1):
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
#
        self.fc_layers = nn.Sequential(*layers)

    #------------------------
    def forward(self, x):
        #print("ffnn, enter forward, x: ", x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc_layers(x)
        #print("ffnn, after fc_layers, x.shape: ", x.shape)

        self.out_shape = x.shape
        if self.datum_out_shape != None:
            x = x.view(x.shape[0], *self.datum_out_shape[1:])

        #print("ffnn, x.shape: ", x.shape)
        return x

#----------------------------------------------------------------------
