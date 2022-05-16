#!/usr/bin/env python
# coding: utf-8

# (nonlinearity:exercise)=
# ### Von-May function
# 
# **Task 1**: Write a function which solves the Von-May-Equation.
# 
# 
# **Problem description:**
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
    # Write your code here
    pass


# **Task 2:** Run the code for several initial and parameter combination. What is particularly striking about increasing r-values?
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


# **Extend the Von-May function**
# 
# **Task 3:** Extend this Von-May function by generating 20 random r-values and run simulations with them. Sample the values from a normal distribution with mean 3.95 and standard deviation 0.015 (limit the r-values between 0 and 4). Then average over all time series. Plot both the time series, the averaged time series and the histogram of the averaged time series. What do you observe?

# In[3]:


import random
import numpy as np

def ensemble_may(n, y0, r):
    # Write your code here
    pass


# In[4]:


# Plot the results


# **Revisit the EBM-Model**
# 
# Include a dynamic transmissivity in the energy balance model.
# 
# **Task 4:** Run the energy balance model $T(0)=288 ~ K$, $C_w= 2\cdot10^8 ~ J/(m^2
#  57 \cdot K)$, $\alpha=0.3$, and $\tau_{mean}=0.608 (\pm 10\%)$

# In[5]:


import random
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
    # Write your code here
    pass


def ebm_stochastic(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.64, years=100):
    # Write your code here
    pass
        


# In[6]:


# Plot the results


# **Extend the model with a simple ice/land use albedo parameterisation. (sigmoid version)**
# 
# **Task 5:** In this parameterisation, the albedo is solely a function of mean temperature. As a non-linear function we assume a sigmoid function with
# 
# \begin{align}
# \alpha(T_i) = 0.3 \cdot (1-0.2 \cdot \tanh(0.5 \cdot (T_i-288))).
# \end{align}
# 
# Run the energy balance model for 100 years with four different initial conditions for T(0)=286.0, 287.9, 288.0, and 293.0 K, while fixing the other parameters to $C_w$= 2$\cdot10^8$ J/(m$^2 \cdot$ K) and $\tau_{mean}$=0.608. 
# 
# What can be said about the state of equilibrium?

# In[7]:


import random
import numpy as np
import matplotlib.pyplot as plt


def ebm_ice_albedo(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.64, years=100):
    # Write your code here
    pass


# In[8]:


# Plot the albedo function


# In[9]:


# Run the simulations and plot the results


# **Extend the model with a simple ice/land use albedo parameterisation. (linear version)**
# 
# **Task 6:** In this parameterisation, the albedo is solely a function of mean temperature. We assume a simple linear relation according to
# 
# \begin{align}
#     f(x)= 
# \begin{cases}
#     \alpha_i,& \text{if } T\leq T_i\\
#     \alpha_g,& \text{if } T \geq T_g\\
#     \alpha_g+b(T_g-T) & \text{if } T_i<T<T_g
# \end{cases}
# \end{align}
# 
# with $T_i$=273 K, and $T_g$= 292 K. Run the energy balance model for 100 years with four different initial conditions for T(0)=286.0, 287.9, 288.0, and 293.0 K, while fixing the other parameters to $C_w$= 2$\cdot10^8$ J/(m$^2 \cdot$ K), and $\tau_{mean}$=0.608, $a_i$=0.6, and $a_g$=0.2. 
# 
# What can be said about the state of equilibrium?

# In[10]:


import random
import numpy as np
import matplotlib.pyplot as plt


def ebm_ice_albedo_2(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.608, years=100):
    # Write your code here
    pass


# In[11]:


# Run the simulations and plot the results


# **Task 7:** Determine the equilibrium climate sensitivity (ECS) and the feedback factor for the simulation from Task 5 using T(0)=289 K.  (sigmoid albedo parametrisation)

# In[12]:


import random
import numpy as np
import matplotlib.pyplot as plt


def ebm_ice_albedo_stochastic_ECS(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.64, years=100):
    # Write your code here
    pass


# In[13]:


# Run the simulations and plot the results


# **Task 8:** Repeat the simulation from Task 5 (sigmoid function for albedo) with T(0)=289 K, but again sample the transmissivity from a normal distribution with a standard deviation of 10%.  
# 
# What special feature can now be observed? What conclusions can be inferred regarding the prediction of weather and climate?

# In[14]:


import random
import numpy as np
import matplotlib.pyplot as plt


def ebm_ice_albedo_stochastic(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.64, years=100):
    # Write your code here
    pass


# In[15]:


# Plot the results


# In[16]:


# Make more plots to illustrate the characteristics of the time series

