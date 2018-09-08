import numpy as np
from IPython import embed
import glob
import sys

recon_data = []
class_data = []
recon_class_data = []
d500_files = glob.glob('results/data_500/*')
d100_files = glob.glob('results/data_first_pass/*')

for folder in [d500_files, d100_files]:
    for full_fname in folder:
        fname = full_fname.split('/')[-1]
        first = fname.split('_')[0]
        last = fname.split('_')[-1]
        if first == 'data':
            if last == 'recon':
                recon_data.append(np.load(full_fname))
            if last == 'class' and fname.split('_')[-2] == 'recon':
                recon_class_data.append(np.load(full_fname))

np.save('results/recon_data.npy', np.array(recon_data))
np.save('results/reconclass_data.npy', np.array(recon_class_data))


