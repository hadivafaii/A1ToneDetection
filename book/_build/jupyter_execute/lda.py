#!/usr/bin/env python
# coding: utf-8

# # Linear Discriminant Analysis (LDA)
# 
# The main motivation behind LDA is to find a lower dimensional projection of data that maximizes **between class** distances, while minimizing **within class** distances of data points.  This dimensionality reduction will lead to a maximally separable projection of the data, and in a sense, one can consider this as supervised PCA.  The method is described in chapter 4 of the the famous [ESL textbook](https://web.stanford.edu/~hastie/Papers/ESLII.pdf).
# 
# For each experiment, the goal is to find a low-dimensional ($d=3$) space that maximally separates either {**hit, miss, correctreject,** and **falsealarm**} trials, or {**target7k, target14k, nontarget14k,** and **nontarget20k**} trials. This way, we can visualize trajectories to see check whether they diverge as time goes on. Furthermore, the resulting projection of features down to this smaller space can be used to learn a linear classifier.  Results show that we can indeed learn a low-$d$ space where trial trajectories diverge, while maintaining a high classification score as well.  Results obtained using [sklearn's implementation of LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).

# ## Imports

# In[1]:


import os
import sys
sys.path.insert(0, '/home/hadivafa/Dropbox/git/A1ToneDiscrimination/')
from utils.plot_functions import *
from analysis.lda_analysis import LDA

import numpy as np
import pandas as pd
from datetime import datetime
from os.path import join as pjoin
from copy import deepcopy as dc
from tqdm import tqdm
import pickle
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


# ## Dirs

# In[2]:


base_dir = pjoin(os.environ['HOME'], 'Documents/Kanold')
processed_dir = pjoin(base_dir, 'python_processed')
results_dir = pjoin(base_dir, 'results', 'lda')

nb_std = 1
h_load_file = pjoin(processed_dir, "organized_nb_std={:d}.h5".format(nb_std))


# ## 1) 4way classification
# 
# First, we consider only: **{hit, miss, correctreject, falsealarm}**

# ### Performance comparison figure:
# 
# We first compare LDA results obtained by projecting down to a lower dimensional space of differenct sizes.  Because of the way LDA is defined, we are limited to $num\_dims <= num\_classes - 1$ which leaves us with options $dim \in [1, 2, 3]$.  For the 4way classification between different behavioral trials types, we see a slight increase in performance as we increase number of dims.  This increase in performance is not large enough to have significant implications. As another evaluation measure, we next look at Sb, the scatter matrix between classes. Sb is largest for the case of $dim =3$ which means using $dim = 3$ will result in more separable trajectories.  Overall, these results show that for the case of 4way classification, and in the context of LDA, choosing a 3-dimnensional space works better.

# In[3]:


load_dir = pjoin(results_dir, '4way')

_ = mk_lda_summary_plot(load_dir, dpi=70)


# ### Trajectory figures:
# 
# Now let's look at some trajectories.
# 
# - There are 3 plots on top row:
#     - The one on the left shows average performance for the 4 way classification task
#     - The one in the middle is the average distance between data points from different classes
#     - The one on the right is scatter matrix between classes (similar to mid, but it is more standard when using LDA)
# - Then at the bottom there is a representative trajecory:
#     - As we can see, these 4 trajectories diverge and they reach maximum separation at best timepoint
#     - The trajectories show how center-of-mass (COM) of each class evolves in time
#     - The half-transparent shapes around each scatter point accounts for the variance (i.e. the size of the 'cloud' around COM).  Larger = more variance, therefore, if datapoints were all on top of eachother at the COM, it would have zero redius.

# #### 3D:

# In[4]:


_ = mk_trajectory_plot(
    load_dir,
    dim=3,
    global_stats=True,
    figsize=(10, 12),
    dpi=60,
)


# #### 3D, Shuffled:

# In[5]:


name = 'lana_2018-08-17'

_ = mk_trajectory_plot(
    load_dir,
    dim=3,
    name=name,
    global_stats=False,
    shuffled=True,
    figsize=(10, 12),
    dpi=60,
)


# ### GIFs that rotate for a better view
# 
# (temporary, later I will use plotly for interactive figures)

# In[6]:


from IPython.display import Image
with open('traj_azim.gif','rb') as file:
    display(Image(file.read(), width=600, height=500, embed=True))


# In[7]:


from IPython.display import Image
with open('traj_rot.gif','rb') as file:
    display(Image(file.read(), width=600, height=500, embed=True))


# ### Lower dimensions:

# #### 2D:

# In[8]:


_ = mk_trajectory_plot(
    load_dir,
    dim=2,
    global_stats=True,
    shuffled=False,
    figsize=(10, 12),
    dpi=60,
)


# #### 2D, Shuffled:

# In[9]:


_ = mk_trajectory_plot(
    load_dir,
    dim=2,
    global_stats=False,
    shuffled=True,
    figsize=(10, 10),
    dpi=60,
)


# #### 1D:

# In[10]:


_ = mk_trajectory_plot(
    load_dir,
    dim=1,
    global_stats=True,
    shuffled=False,
    figsize=(10, 8),
    dpi=60,
)


# #### 1D, Shuffled:

# In[11]:


name = "gabby_2016-08-21"

_ = mk_trajectory_plot(
    load_dir,
    dim=1,
    name=name,
    global_stats=False,
    shuffled=True,
    figsize=(10, 8),
    dpi=60,
)


# ### Discussion:
# 
# So far, we saw that projecting neural responses down to a lower dimensional space found by LDA revels diverging trajectories.  But what is the content of this lower dimensional space?  Before addressing this quesiton, let us now look at another classification task between different target and nontarget frequencies.  After that we will come back to the interpretability question.

# ## 2) StimFrequency classification
# 
# Now we consider target/nontarget related labels: **{target7k, target10k, nontarget14k, nontarget20k}**
# 
# Just like above, there is no meaningful different in performance between different dimensionalities, but trajectories diverge beautifully.

# ### Performance figure

# In[12]:


load_dir = pjoin(results_dir, 'stimfreq')

_ = mk_lda_summary_plot(load_dir, dpi=70)


# ### Trajectory figures

# #### 3D:

# In[13]:


_ = mk_trajectory_plot(
    load_dir,
    dim=3,
    global_stats=True,
    shuffled=False,
    figsize=(10, 12),
    dpi=60,
)


# #### 3D, Shuffled:

# In[14]:


name = "gabby_2016-08-18"

_ = mk_trajectory_plot(
    load_dir,
    dim=3,
    name=name,
    global_stats=False,
    shuffled=True,
    figsize=(10, 12),
    dpi=60,
)


# #### 2D:

# In[15]:


_ = mk_trajectory_plot(
    load_dir,
    dim=2,
    global_stats=True,
    shuffled=False,
    figsize=(10, 12),
    dpi=60,
)


# #### 2, Shuffled:

# In[16]:


name = "gabby_2016-08-18"

_ = mk_trajectory_plot(
    load_dir,
    dim=2,
    name=name,
    global_stats=False,
    shuffled=True,
    figsize=(10, 10),
    dpi=60,
)


# #### 1D:

# In[17]:


_ = mk_trajectory_plot(
    load_dir,
    dim=1,
    global_stats=True,
    shuffled=False,
    figsize=(10, 8),
    dpi=60,
)


# #### 1D, Shuffled:

# In[18]:


name = "gabby_2016-08-18"

_ = mk_trajectory_plot(
    load_dir,
    dim=1,
    name=name,
    global_stats=False,
    shuffled=True,
    figsize=(10, 8),
    dpi=60,
)


# In[ ]:





# In[ ]:





# In[ ]:





# ## To be continued...

# In[ ]:





# In[ ]:




