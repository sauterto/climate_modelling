#!/usr/bin/env python
# coding: utf-8

# (greenhouse:exercise)=
# # Greenhouse model

# **Task 1**: Plug Eq. (7) into Eq. (6) and solve for the radiative equilibrium suface temperature $T_e$ 

# In[1]:


# Solve for the radiative equilibrium temperature Te
# Put you own code here
Te = 0

print('Radiative equilibrium temperature: {:.2f}'.format(Te))


# **Task 2**: Where in the atmosphere do we find $T_e$?

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


## NCAR data
url = 'air.mon.ltm.1981-2010.nc'


# In[4]:


#  Take global, annual average and convert to Kelvin


# In[5]:


#  a "quick and dirty" visualization of the data


# In[6]:


# Create the skew-plot with the metpy module


# **Task 3**: What is the surface temperature with the single layer model? 

# In[7]:


# Solve for the atmospheric surface temperature

# Calc surface temperature Ts
# Put you own code here
Ts = 0

print('Surface temperature: {:.2f}'.format(Ts))


# Why does the model overestimate the surface temperature?

# **Task 5**: Write a Python function for $OLR = U_2 = (1-\epsilon)^2 \sigma T_s^4 + \epsilon(1-\epsilon)\sigma T_0^4 + \epsilon \sigma T_1^4$

# In[8]:


def two_layer_model(Ts, T0, T1, epsilon):
    pass


# **Task 7**: We will tune our model so that it reproduces the observed global mean OLR given observed global mean temperatures. Determine the temperatures for the two-layer model from the following sounding

# ![alt text](pics/vertical_profile.png "Sounding")

# **Task 8**: Find graphically the best fit value of $\epsilon$
# 

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

# Write your code here


# In[10]:


# Validate the result


# **Task 9**: Write a Python function to calculate each term in the OLR. Plug-in the observed temperatures and the tuned value for epsilon.

# In[11]:


def two_layer_terms(Ts, T0, T1, epsilon):
    pass


# In[12]:


# Calculate terms


# **Task 10**: Changing the level of emission by adding absorbers, e.g. by 10 %. 
# Suppose further that this increase happens abruptly so that there is no time for the temperatures to respond to this change. We hold the temperatures fixed in the column and ask how the radiative fluxes change.
# 
# Which terms in the OLR go up and which go down?

# In[13]:


# Make simulation here


# **Task 11**: Calculate the radiative forcing for the previous simulation

# In[14]:


# Calculate radiative forcing


# **Task 12**: What is the greenhouse effect for an isothermal atmosphere?

# In[15]:


# Make simulation here


# **Task 13**: For a more realistic example of radiative forcing due to an increase in greenhouse absorbers, we use our observed temperatures and the tuned value for epsilon. Assume an increase of epsilon by 2 %.

# In[16]:


# Make simulation here

