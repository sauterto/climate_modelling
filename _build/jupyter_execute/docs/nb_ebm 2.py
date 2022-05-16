#!/usr/bin/env python
# coding: utf-8

# (ebm:exercise)=
# # Simplified Energy Balance Model

# **Task 1:** Write a Python function for the OLR and effective temperature for later use.

# In[1]:


# Stefan-Boltzmann constant
sigma = 5.67e-8

def T_eff(OLR):
    """ Effective global mean emission temperature """
    # Write your code here
    pass
     
def OLR(T):
    """ Stefan-Boltzmann law """
    # Write your code here
    pass


# **Task 2:** Extend the OLR function by another **transmissivity** constant $\tau$ which takes this effect into account. Determine the transmissivity for a global mean temperature of 288 K.

# In[2]:


def OLR(T, tau):
    """ Stefan-Boltzmann law """
    # Write your code here
    pass

def tau(OLR, T):
    """ Calculate transmissivity """
    # Write your code here
    pass


# In[3]:


# Print the results


# **Task 3:** Determine the planetary albedo from the observations and write a function for the absorbed shortwave radiation, the part of the incoming sunlight that is not reflected back to space

# In[4]:


# Calculate the planetary albedo



# **Task 4:** What additional amount of energy would have to remain in the system for the global temperature to rise by 4 K?

# In[5]:


def ASR(Q, albedo):
    """ Absorbed shortwave radiation """
    # Write your code here
    pass    # Write your code here
    pass


# In[6]:


# Print the results


# **Task 5:** Rearrange the equation according to the temperature denoting our equilibrium temperature. Substitute the observed values for insolation, transmissivity and planetary albedo and calculate the equlibrium temperature.

# In[7]:


def equilibrium_temperature(alpha,Q,tau):
    """ Equilibrium temperature """
    # Write your code here
    pass


# In[8]:


# Print the results


# **Task 6:** With simple approaches such as equlibrium temperature, conceptual scenarios can be calculated. For example, the connection between the increase in albedo due to more cloud cover and the associated decrease in transmissivity can be investigated. For example, assume that the planetary albedo increases to 0.32 due to more cloud cover and that the transmissivity decreases to 0.57. What is the equilibrium temperature?

# In[9]:


# Make your calculations here


# **Task 8:** Write a function called *step_forward(T, dt)* that returns the new temperature given the old temeprature T and timestep dt. Assume an initial temperature of 288 K and integrate the function for a few timestep and observe how the temperature changes.

# In[10]:


def step_forward(Q, T, Cw, alpha, tau, dt):
    # Write your code here
    pass


# In[11]:


# Do first step forward


# In[12]:


# Do second step forward


# In[13]:


# Do third step forward


# **Task 9:** Integrate the equation over a time of 200 years and plot the result. Use the following initial and boundary conditions: 
# 
# $
# \begin{align}
# S_0 &=1360 ~ W m^{-2} \\
# T_0 &= 273 ~ K \\ 
# C_w &= 10^8 ~ J/(m^2 \cdot K) \\
# \alpha &= 0.3 \\
# \tau &= 0.64
# \end{align}
# $

# In[14]:


import numpy as np
import matplotlib.pyplot as plt


def OLR(T, tau):
    """ Stefan-Boltzmann law """
    # Write your code here
    pass

def ASR(Q, alpha):
    """ Absorbed shortwave radiation """
    # Write your code here
    pass


def step_forward(Q, T, Cw, alpha, tau, dt):
    """ Time integration """
    # Write your code here
    pass


def ebm(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.64, years=100):
    ''' This is a simple Energy Balance Model with global radiation and outgoing longwave radiation.'''
     # Write your code here
    pass




# In[15]:


# Experiment 1


# **Task 10:** What happens if the intial temperature is set to 293 K ?

# In[16]:


# Experiment 2


# **Task 11:** What changes do you observe with a higher $C_w$ value (e.g. $C_w=10\cdot10^8 ~ J/(m^2 \cdot K)$)?
# 

# In[17]:


# Experiment 3


# **Task 12:** How does the result change when $\tau=1$?

# In[18]:


# Experiment 4


# ### Case Study: Venus. 
# 
# Calculate the mean surface temperature on Venus. Due to its proximity to the Sun, Venus has a very high irradiance of $S_{0}=2619 ~ Wm^{-2}$. Due to the high cloud cover, the albedo is about 0.7. What surface temperature can be expected? (Use the previous values for $C_w$ and $\tau$).

# In[19]:


# Experiment 5


# Compare the measurements with your simulation. 
# 
# Is there a difference? If so, why does this difference exist? (Use the model to prove your hypothesis)

# In[20]:


# Experiment 6


# In[ ]:




