#!/usr/bin/env python3

import pickle
from sklearn.metrics import roc_auc_score as ras
import scipy.ndimage as snd
import numpy as np
from mcmc_lumpy import *

# load mcmc results
with open('ratios.pkl','rb') as f:
    lr, phi_set = pickle.load(f)

# Get the imagess for each of the phi sets
g = []
for phi in phi_set:
    g.append(phi.grab_g().ravel())

# Calculate HO results for each background
# with signal 0.1 and noise 0.01
ho = []
obj_dim1 = [28, 33]
obj_dim2 = [29, 32]
signal = np.zeros((64,64))
signal[obj_dim1[0]:obj_dim1[1], obj_dim2[0]:obj_dim2[1]] = 0.1
signal[obj_dim2[0]:obj_dim2[1], obj_dim1[0]:obj_dim1[1]] = 0.1
signal = snd.filters.gaussian_filter(signal, 2).ravel()
for img in g:
    ho.append(np.dot(signal,img))
print(ras(100*[1,0],ho))
print(ras(100*[1,0],lr))
