#!/usr/bin/env python
# coding: utf-8

# In[1]:


sigma = 5.67e-8
Q = 342
albedo = 0.3

Te = (((1-0.3)*Q)/sigma)**(1/4)
Te


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


# In[3]:


ncep = xr.open_dataset('./files/air.mon.ltm.1981-2010.nc', 
                       use_cftime=True)


# In[4]:


ncep


# In[5]:


# calculate the area-weighted temperature over its domain. This dataset has a regular latitude/ longitude grid, 
# thus the grid cell area decreases towards the pole. For this grid we can use the cosine of the latitude as proxy 
# for the grid cell area.
weights = np.cos(np.deg2rad(ncep.lat))

# Use the xarray function to weight the air temperature array
air_weighted = ncep.air.weighted(weights)

# Take the mean over lat/lon/time to get a mean vertical profile
weighted_mean = air_weighted.mean(("lat","lon", "time"))


# In[6]:


weighted_mean


# In[7]:


# Import the metpy library
from metpy.plots import SkewT


# In[8]:


fig = plt.figure(figsize=(15,15))
skew = SkewT(fig, rotation=30)
skew.plot(weighted_mean.level, weighted_mean, color='black', linestyle='-')

skew.plot_dry_adiabats()
skew.plot_moist_adiabats()


# In[9]:


Ts = 2**(1/4) * Te
Ts


# In[10]:


def two_layer_model(Ts, T0, T1, epsilon):
    sigma = 5.68e-8
    return ((1-epsilon)**2)*sigma*Ts**4 + \
        epsilon*(1-epsilon)*sigma*T0**4 + \
        epsilon*sigma*T1**4


# In[11]:


two_layer_model(288, 270, 250, 0.6)


# In[12]:


OLR = []
epsilons = []
OLR_obs = 238.5

def find_nearest(array, value):
    idx, val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

for eps in np.arange(0, 1, 0.01):
    OLR.append(OLR_obs - two_layer_model(288.0, 275.0, 230, eps))
    epsilons.append(eps)
    
# Find the closest value
idx = find_nearest(OLR, 0)

# Save the optimized epsilon
epsilon = epsilons[idx]

print('The optimized epsilon is: {:.2f}'.format(epsilons[idx]))


# In[13]:


plt.figure(figsize=(15,8))
plt.plot(epsilons, OLR)
plt.scatter(epsilons[idx], OLR[idx], s=50, color='r')
plt.hlines(0,0,1,linestyle='dotted',color='gray')


# In[14]:


two_layer_model(288,275,230,0.59)


# In[15]:


def two_layer_term(Ts, T0, T1, epsilon):
    sigma = 5.68e-8
    return ((1-epsilon)**2)*sigma*Ts**4, \
        epsilon*(1-epsilon)*sigma*T0**4, \
        epsilon*sigma*T1**4


# In[16]:


term1, term2, term3 = two_layer_term(288, 288, 288, 0.59)


# In[17]:


print('Term 1: {:.2f}'.format(term1))
print('Term 2: {:.2f}'.format(term2))
print('Term 3: {:.2f}'.format(term3))
print('Total: {:.2f}'.format(term1+term2+term3))


# In[ ]:




