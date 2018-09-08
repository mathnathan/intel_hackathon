import numpy as np
import subprocess

from tqdm import tqdm
mode = 0

num_points = 500
num_modes = 2
max_current_processes = 10

assert num_points%num_modes == 0


for m in range(num_modes):
    for p in tqdm(range(num_points//max_current_processes)):
        for i in range(max_current_processes):

            if i == max_current_processes-1:
                subprocess.Popen(["python vae_mnist.py {} {} {}".format(p,i,m)], shell=True).wait()
            else:
                proc = subprocess.Popen(["python vae_mnist.py {} {} {}".format(p,i,m)], shell=True)#,


                 #stdin=None, stdout=None, stderr=None, close_fds=True)


