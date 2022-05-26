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


Tglobal


# In[27]:


# Import the metpy library
from metpy.plots import SkewT


# In[40]:


fig = plt.figure(figsize=(15,15))
skew = SkewT(fig, rotation=30)
skew.plot(weighted_mean.level, weighted_mean, color='black', linestyle='-')

skew.plot_dry_adiabats()
skew.plot_moist_adiabats()


# In[71]:


array = [1,2,3,0]
value = 3

idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
print(idx)


# In[70]:


a = lambda x: abs(x[1]-value)
for idx in enumerate(array):
    print(idx[1])
    


# In[ ]:




