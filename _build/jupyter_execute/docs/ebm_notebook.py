#!/usr/bin/env python
# coding: utf-8

# <img src="pics/ebm_02.jpg" width="600" >

# In[1]:


# Stefan-Boltzmann constant
sigma = 5.67e-8

def T_eff(OLR):
    """ Effective global mean emission temperature """
    return (OLR/sigma)**(0.25)
     
def OLR(T):
    """ Stefan-Boltzmann law """
    return sigma * T**4


# In[2]:


def OLR(T, tau):
    """ Stefan-Boltzmann law """
    return tau * sigma * T**4

def tau(OLR, T):
    """ Calculate transmissivity """
    return OLR / (sigma*T**4)


# In[3]:


print("Transmissivity assuming a global mean temperature of 288 K: {}".format(tau(238.5, 288)))
print("Additionaly energy to increase global mean temperature by 4 K: {} W m^-2".format(OLR(292, 0.61)-OLR(288, 0.61)))


# In[4]:


Q = 341.3           # area-averaged insolation 
Freflected = 101.9  # reflected shortwave flux in W/m2
alpha = Freflected/Q

print("Planetary Albedo: {0}".format(alpha))


# In[5]:


def ASR(Q, albedo):
    """ Absorbed shortwave radiation """
    return (1-albedo) * Q


# In[6]:


print("Absorbed shortwave radiation: {}".format(ASR(Q, alpha)))


# In[7]:


def equilibrium_temperature(alpha,Q,tau):
    """ Equilibrium temperature """
    return ((1-alpha)*Q/(tau*sigma))**(1/4)

Teq_observed = equilibrium_temperature(alpha,Q,tau(238.5, 288))
print(Teq_observed)


# In[8]:


Teq_new = equilibrium_temperature(0.32,Q,0.57)

#  an example of formatted print output, limiting to two or one decimal places
print('The new equilibrium temperature is {:.2f} K.'.format(Teq_new))
print('The equilibrium temperature increased by about {:.1f} K.'.format(Teq_new-Teq_observed))


# ### Task 2: Write a function ebm which solves the energy balance equation.

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

def ebm(SWin,T0,c,alpha,tau,years):
    ''' This is a simple Energy Balance Model with global radiation and outgoing longwave radiation.

    Syntax: ebm(T0,c,alpha,tau)

    with
    T0     :: Initial temperature (Kelvin)
    c      :: Heat capacity J/(m2*K)
    alpha  :: Albedo [-]
    tau    :: Transmissivity [-]

    Example: ebm(273.2, 10e8, 0.3, 0.64)

    Author: Tobias Sauter
    Date:   06/04/2022
    '''

    # Constants
    epsilon = 5.67e-8;    # Stefan-Bolzmann Constant (W/(m2*K4);

    # Time step
    steps = 10;                           # days
    dt = 60*60*24*steps;                  # convert days to seconds
    integration = (365/steps)*years;      # Integrate over x years

    # Init arrays and variables
    timeseries = [];
    Ti = T0;

    # Calculation
    for z in range(int(integration)):
        T = Ti + (dt/c * (SWin - alpha*SWin - 0.95*epsilon*Ti**4*tau));  
        timeseries.append(T);
        Ti = T;
        
    return np.array(timeseries)



# ### Task 3: Integrate the equation over a time of 200 years and plot the result. Use the following initial and boundary conditions: 
# 
# $
# \begin{align}
# S_0 &=1360 ~ W m^{-2} \\
# T(0)&= 273 ~ K \\ 
# C_w &= 10^8 ~ J/(m^2 \cdot K) \\
# \alpha &= 0.3 \\
# \tau &= 0.64
# \end{align}
# $

# In[10]:


# Integrate the model
T_273 = ebm(342.0,273.2, 10e8, 0.3, 0.64, 500)

# Plot results
fig = plt.figure(figsize=(20,5))
plt.plot(T_273)


# ### Task 4: What happens if the intial temperature is set to 293 K ?

# In[11]:


# Integrate the model
T_293 = ebm(342.0,293.2, 10e8, 0.3, 0.64, 500)

# Plot results
fig = plt.figure(figsize=(20,5))
plt.plot(T_293)
plt.plot(T_273)


# ### Task 5: What changes do you observe with a higher $C_w$ value (e.g. $C_w=10\cdot10^8 ~ J/(m^2 \cdot K)$)?
# 

# In[12]:


# Integrate the model
T_293_Cw = ebm(342.0,293.2, 10*10e8, 0.3, 0.64, 500)

# Plot results
fig = plt.figure(figsize=(20,5))
plt.plot(T_293)
plt.plot(T_273)
plt.plot(T_293_Cw)


# ### Task 6: How does the result change when $\tau=1$?

# In[13]:


# Integrate the model
T_293_tau = ebm(342.0,293.2, 10e8, 0.3, 1.0, 500)

# Plot results
fig = plt.figure(figsize=(20,5))
plt.plot(T_293, label='$T_{293}$')
plt.plot(T_273, label='$T_{273}$')
plt.plot(T_293_Cw, label='$T_{293 C_w}$')
plt.plot(T_293_tau, label='$T_{293 \tau}$')

plt.legend()


# ### Case Study: Venus. 
# 
# Calculate the mean surface temperature on Venus. Due to its proximity to the Sun, Venus has a very high irradiance of $S_{0}=2619 ~ Wm^{-2}$. Due to the high cloud cover, the albedo is about 0.7. What surface temperature can be expected? (Use the previous values for $C_w$ and $\tau$).

# In[14]:


# Integrate the model
T_venus = ebm(2619/4, 288.2, 10e8, 0.7, 0.64, 500)

# Plot results
fig = plt.figure(figsize=(20,5))
plt.plot(T_venus, label='$T_{venus}$')
plt.legend()


# ### Compare the measurements with your simulation. 
# 
# Is there a difference? If so, why does this difference exist? (Use the model to prove your hypothesis)

# In[15]:


# Integrate the model
T_venus = ebm(2619/4, 288.2, 10e8, 0.7, 0.015, 500)

# Plot results
fig = plt.figure(figsize=(20,5))
plt.plot(T_venus, label='$T_{venus}$')
plt.legend()


# In[ ]:




