#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
plt.ion()

import Corrfunc

from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi


# # Tophat on theory data

# ### Set up parameters
# 
# Here we use a low-density lognormal simulation box.

# In[2]:


boxsize = 750
nbar_str = '1e-5'
proj_type = 'tophat'
#proj_type = None

rmin = 40
rmax = 150
nbins = 11

mumax = 1.0
seed = 0
#weight_type='pair_product'
weight_type=None


# In[3]:


rbins = np.linspace(rmin, rmax, nbins+1)
rcont = np.linspace(rmin, rmax, 1000)

cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
cat_dir = '../byebyebias/catalogs/cats_lognormal{}'.format(cat_tag)

periodic = False
cosmo = 1 #doesn't matter bc passing cz, but required
nthreads = 24
nmubins = 1
verbose = True


# ### Load in data and randoms

# In[4]:


# data
data_fn = '{}/cat_lognormal{}_seed{}.dat'.format(cat_dir, cat_tag, seed)
data = np.loadtxt(data_fn)
x, y, z = data.T
nd = data.shape[0]
#weights = np.full(nd, 0.5)
weights = None


# In[5]:


# randoms
rand_fn = '{}/rand{}_10x.dat'.format(cat_dir, cat_tag)
random = np.loadtxt(rand_fn)
x_rand, y_rand, z_rand = random.T
nr = random.shape[0]
#weights_rand = np.full(nr, 0.5)
weights_rand = None


# In[6]:


def extract_counts(res, weight_type=None):
    counts = np.array([x[4] for x in res], dtype=float)
    print(counts)
    if weight_type:
        weights = np.array([x[5] for x in res], dtype=float)
        print(weights)
        counts *= weights
    return counts


# In[7]:


# standard
dd_res_corrfunc, _, _ = DDsmu(1, nthreads, rbins, mumax, nmubins, x, y, z,
                proj_type=proj_type, nprojbins=nbins,
                verbose=verbose, boxsize=boxsize)
dd = extract_counts(dd_res_corrfunc, weight_type)

# dr_res_corrfunc, _, _ = DDsmu(0, nthreads, rbins, mumax, nmubins, 
#                 x, y, z, X2=x_rand, Y2=y_rand, Z2=z_rand, 
#                 proj_type=proj_type, nprojbins=nbins, 
#                 verbose=verbose,
#                 weights1=weights, weights2=weights_rand, weight_type=weight_type)
# dr = extract_counts(dr_res_corrfunc, weight_type)

# rr_res_corrfunc, _, _ = DDsmu(1, nthreads, rbins, mumax, nmubins, x_rand, y_rand, z_rand,
#                 proj_type=proj_type, nprojbins=nbins,
#                 verbose=verbose,
#                 weights1=weights_rand, weight_type=weight_type)
# rr = extract_counts(rr_res_corrfunc, weight_type)


# In[ ]:


#!jupyter nbconvert --to script example.ipynb

