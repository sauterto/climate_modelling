#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

Km = 0.1
Kh = 0.01
g = 9.81
lapse = 0.004
s = -5
s = np.sin(s*3.14/180)
C = -5
T0 = 280
zmax = 100


lam = ((4*T0*Km*Kh)/(g*s**2*lapse))**(0.25)
mu = ((g*Kh)/(T0*Km*lapse))**(0.5)

theta1 = [C*np.exp(-z/lam)*np.cos(-z/lam) for z in np.arange(0,zmax,0.5)]
u1 = [C*mu*np.exp(-z/lam)*np.sin(-z/lam) for z in np.arange(0,zmax,0.5)]

lam2 = ((g*s**2*lapse)/(4*T0*Km*Kh))**(0.25)

theta3 = [C*np.exp(-z*lam2)*np.cos(-z*lam2) for z in np.arange(0,zmax,0.5)]
#theta2 = [C*np.exp(-z*lam2)*np.cos(-z*lam2) for z in np.arange(0,zmax,0.5)]
#u2 = [C*np.exp(-z*lam2)*np.sin(-z*lam2) for z in np.arange(0,zmax,0.5)]

print(np.max(np.array(u1)-np.array(u2)))
print(np.max(np.array(theta1)-np.array(theta2)))
print(1/lam,lam2)

plt.plot(theta1,np.arange(0,zmax,0.5))
plt.plot(u1,np.arange(0,zmax,0.5))

plt.plot(theta2,np.arange(0,zmax,0.5))
plt.plot(u2,np.arange(0,zmax,0.5))


# In[ ]:


np.arange(0,10,0.1)


# In[ ]:




