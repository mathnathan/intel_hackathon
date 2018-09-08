import numpy as np
import sys
from tqdm import tqdm
from IPython import embed
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances as pd
import matplotlib.pyplot as plt

rd = np.load('results/recon_data.npy')
rcd = np.load('results/reconclass_data.npy')
ut_indices = np.triu_indices(1000,k=1)
fingerprints = []
labels = []
for rep in tqdm(rd):
    fingerprints.append(pd(rep)[ut_indices])
    labels.append(0)
for rep in tqdm(rcd):
    fingerprints.append(pd(rep)[ut_indices])
    labels.append(1)

tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=400)
compressed_fingerprints = tsne.fit_transform(fingerprints)
plt.scatter(compressed_fingerprints[:,0], compressed_fingerprints[:,1], c=labels)

plt.show()


embed()

