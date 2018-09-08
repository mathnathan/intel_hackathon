import input_data as g  # global variables
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import os, sys
import numpy as np
import utils as u
from IPython import embed


class DNN(nn.Module):

    def __init__(self, params):
        """Constructor for the simple feed forward architecture.

            params['input_shape']
                The shape of the input being passed into the network.
                For now, must include everything including batch size.

            params['transfer_funcs']
                A tuple or a single function. If a tuple, must specify
                a func for each layer. Otherwise the single function is
                duplicated for each layer. If no function is specified
                it defaults to a linear activation.

            params['architecture']
                A tuple of numbers specifying the number
                of nodes in each layer of the feed forward network. The tuple
                *must* explicitly state the number of nodes in each layer.

            Example:

                To create a network with 4 layers of size 128, 128, 64, 32
                where each layer uses an exponential linear unit activation
                function with alpha set to 0.9, that operates on input of
                shape 256 with batch size 100 use the following parameters

                params['input_shape'] = (100, 256)
                params['architecture'] = [128,128,64,32]
                params['transfer_funcs'] = torch.nn.ELU(alpha=0.9)
        """
        super(DNN, self).__init__()

        # The below is hardly error proof. It is just a few minor checks to help
        # with debugging. Eventually it will need to be bolstered further (check
        # architecture for valid entires too, i.e. no negatives etc)
        try:
            self.input_shape = params['input_shape']
            self.architecture = params['architecture']
        except KeyError as key:
            raise KeyError('%s must be specified in params' % (key)) from key

        try:
            self.transfer_funcs = params['transfer_funcs']
        except KeyError as key:
            # Default to Linear activation function
            self.transfer_funcs = lambda x: x

        if hasattr(self.transfer_funcs, '__iter__'): # Check to see if it is iterable
            # If it is iterable that means they put in a list or tuple (hopefully)
            # in which case they must specify an actiavation func for each layer
            ERR_MSG = 'Must specify a transfer function for each layer. i.e. \
            len(architecture) must equal len(transfer_func)'
            assert len(self.architecture) == len(self.transfer_funcs), ERR_MSG
        elif hasattr(self.transfer_funcs, '__call__'): # See if it's a func
            # Then duplicate it for every layer. If the user only puts in one
            # instance, it means they want it for every layer
            #embed(); sys.exit()
            tf = self.transfer_funcs
            del self.transfer_funcs # This is needed for super weird reasons...
            self.transfer_funcs = [tf]*len(self.architecture)
        else:
            sys.exit("transfer_funcs must be an iterable or a function")

        self._layers = nn.ModuleList([])
        self._batchnorm_layers = nn.ModuleList([])
        num_prev_nodes = self.input_shape[1]

        for num_next_nodes in self.architecture:
            self._layers.append(nn.Linear(num_prev_nodes, num_next_nodes))
            self._batchnorm_layers.append(nn.BatchNorm1d(num_next_nodes))
            num_prev_nodes = num_next_nodes

    def forward(self, x):

        for l, f, bn in zip(self._layers, self.transfer_funcs, self._batchnorm_layers):
            x = l(x)
            x = f(x)
            x = bn(x)

        return x
