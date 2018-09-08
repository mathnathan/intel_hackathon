import numpy as np
from IPython import embed
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances as pd
import matplotlib.pyplot as plt

ld0 = np.load('d0.npy')
#ld0 = np.load('results/latent_dim_0.npy')
#tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=400)
#ld0_tsne = tsne.fit_transform(ld0)
#plt.figure(1)
#plt.scatter(ld0_tsne[:,0], ld0_tsne[:,1])

ld1 = np.load('d1.npy')
#ld1 = np.load('results/latent_dim_1.npy')
#tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=400)
#ld1_tsne = tsne.fit_transform(ld1)
#plt.figure(1)
#plt.scatter(ld1_tsne[:,0], ld0_tsne[:,1])
#plt.show()

ut_indices = np.triu_indices(1000,k=1)
d1 = pd(ld0)[ut_indices]
d2 = pd(ld1)[ut_indices]

embed()

