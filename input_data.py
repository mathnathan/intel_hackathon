
img_sz = 64 # square images
sz = img_sz   # size of sub-blocks
num_imgs = 10000
nb_channels = 3

# Convolutional CNN
CNN = 1
DCNN = 1
VAE = 1  # if 0, encoder return mu as opposed to a sampled z

# Hardware
CUDA = False

# for refactor.py
BATCH_SIZE = 50
LOG_INTERVAL = 1
EPOCHS = 1
CUDA = False
LATENT_DIM = 5
FILENAME = 'histograms.npy'


