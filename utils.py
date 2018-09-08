import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

params = np.load("params.npy")[()]
img_width = params['img_sz']
nx_img = params['nx']
ny_img = params['ny']
nb_pix = img_width * nx_img * ny_img

#----------------------------------------------------------------------
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1, input.size(0))
        #return input.view(input.size(0), -1)
#----------------------------------------------------------------------


def plotFigure(rgb_figures):
    # rgb_figures: list of figures to plot in a grid.
    # global parameters: nx_img, ny_img, img_width
    # rgb_fig = np.zeros([nx_img, img_width, ny_img, img_width, 3],dtype='uint8')
    rgb_fig1 = np.zeros([nx_img, ny_img, img_width, img_width, 3], dtype='uint8')

    for i in range(nx_img):
        for j in range(ny_img):
            #fig[i,:,j,:]        =     figures[i+nx_img*j].T
            rgb_fig1[i,j,:,:,:] = rgb_figures[i+nx_img*j].swapaxes(0,1)

    #fig =  fig.reshape(nx_img*img_width, ny_img*img_width).T
    rgb_fig_x = rgb_fig1.swapaxes(1,2).reshape(nx_img*img_width, ny_img*img_width, 3)
    plt.imshow(rgb_fig_x) # works
    plt.show()
#----------------------------------------------------------------------
# Convolution:
def convIO(self, Hin, pad, dil, kernel, stride):
    """
    Hin: input size
    Hout: output size
    Works for any dimension
    """
    #print(pad,dil,kernel,stride)
    Hout = (Hin + 2*pad - dil * (kernel-1) -1) // stride + 1
    return Hout

def deconvIO(self, Hin, padding, out_padding, kernel, stride):
    Hout = ((Hin-1)*stride - 2*padding + kernel + out_padding)
    return Hout

def get_mnist_data(BATCH_SIZE):
    # Load MNIST Data

    mnist_train_data = datasets.MNIST('.', train=True,
        download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train_data,
        batch_size=BATCH_SIZE, shuffle=True)
    mnist_test_data = datasets.MNIST('.', train=False,
        download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_test_data,
        batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader
