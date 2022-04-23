#!/usr/bin/env python
# coding: utf-8

# (nonlinearity:solution)=
# ### Von-May function
# 
# Write a function which solves the Von-May-Equation.
# 
# 
# #### Problem description:
# 
# <blockquote>The starting point for our analysis is the ‘Von-May-Equation’, which is given by <br>
# 
#     
# **\begin{align}    
# y_{t+1} = r \cdot y_{t} \cdot (1-y_{t}),
# \end{align}**
# 
# with  $r$ an pre-defined parameter and $y$ the function value at time $t$ and $t+1$.</blockquote>

# In[1]:


import matplotlib.pyplot as plt

def von_may(y0,r):
    '''
    This function integrates the Von-May Equationn using a then initial condition y0, 
    and the parameter r
    '''
    # Add your code here
    pass 




# Run the code for several initial and parameter combination. What is particularly striking about increasing r-values?
# 
# 
# ```
# y(0)=0.5 and r=2.80 (alternatively, use y(0)=0.9) 
# y(0)=0.5 and r=3.30 (alternatively, use y(0)=0.9) 
# y(0)=0.5 and r=3.95 (alternatively, use y(0)=0.495) 
# y(0)=0.8 and r=2.80 
# 
# ```

# In[2]:


# Integrate the equation and plot the results


# ### Extend the Von-May function
# Extend this Von-May function by generating 20 random r-values and run simulations with them. Sample the values from a normal distribution with mean 3.95 and standard deviation 0.015 (limit the r-values between 0 and 4). Then average over all time series. Plot both the time series, the averaged time series and the histogram of the averaged time series. What do you observe?

# In[3]:


import random
import numpy as np

def ensemble_may(n, y0, r):
    '''
    The function runs n ensemble members of the Von-May-Equation. The function takes the 
    initial condition y0, the parameter r, and the number of ensemble members n.

    '''
    # Add your code here
    pass 



# In[4]:


# Plot the results


# ## Revisit the EBM-Model
# 
# #### Include a dynamic transmissivity in the energy balance model.
# 
# Run the energy balance model $T(0)=288 ~ K$, $C_w= 2\cdot10^8 ~ J/(m^2
#  57 \cdot K)$, $\alpha=0.3$, and $\tau_{mean}=0.608 (\pm 1\%)$

# In[5]:


import random
import numpy as np
import matplotlib.pyplot as plt


def OLR(T, tau):
    """ Stefan-Boltzmann law """
    # Add your code here
    pass 

def ASR(Q, alpha):
    """ Absorbed shortwave radiation """
    # Add your code here
    pass 


def step_forward(Q, T, Cw, alpha, tau, dt):
    # Add your code here
    pass 


def ebm_stochastic(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.64, years=100):
    ''' This is a simple Energy Balance Model with global radiation and outgoing longwave radiation.'''
    # Add your code here
    pass 
        


# In[6]:


# Plot the results


# #### Extend the model with a simple ice/land use albedo parameterisation. 
# 
# In this parameterisation, the albedo is solely a function of mean temperature. As a non-linear function we assume a sigmoid function with
# 
# \begin{align}
# \alpha(T_i) = 0.3 \cdot (1-0.03 \cdot \tanh(1.548 \cdot (T_i-288))).
# \end{align}
# 
# Run the energy balance model for 100 years with four different initial conditions for T(0)=286.0, 288.6, 288.9, and 293.0 K, while fixing the other parameters to $C_w$= 2$\cdot10^8$ J/(m$^2 \cdot$ K), $\alpha$=0.3, and $\tau_{mean}$=0.608. 
# 
# What can be said about the state of equilibrium?

# In[7]:


import random
import numpy as np
import matplotlib.pyplot as plt


def OLR(T, tau):
    """ Stefan-Boltzmann law """
    # Add your code here
    pass 

def ASR(Q, alpha):
    """ Absorbed shortwave radiation """
    # Add your code here
    pass 


def step_forward(Q, T, Cw, alpha, tau, dt):
    # Add your code here
    pass 


def ebm_stochastic(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.64, years=100):
    ''' This is a simple Energy Balance Model with global radiation and outgoing longwave radiation.'''
    # Add your code here
    pass 


# In[8]:


# Plot the results


# #### Repeat the previous simulation, but again sample the transmissivity on a normal distribution with a standard deviation of 3%.  
# What special feature can now be observed? What conclusions can be inferred regarding the prediction of
# weather and climate?
# 

# In[9]:


import random
import numpy as np
import matplotlib.pyplot as plt


def OLR(T, tau):
    """ Stefan-Boltzmann law """
    # Add your code here
    pass 

def ASR(Q, alpha):
    """ Absorbed shortwave radiation """
    # Add your code here
    pass 


def step_forward(Q, T, Cw, alpha, tau, dt):
    # Add your code here
    pass 


def ebm_stochastic(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.64, years=100):
    ''' This is a simple Energy Balance Model with global radiation and outgoing longwave radiation.'''
    # Add your code here
    pass 


# In[10]:


# Plot the results


# In[ ]:




