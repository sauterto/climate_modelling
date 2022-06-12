#!/usr/bin/env python
# coding: utf-8

# (heat_equation:exercises)=
# ### Heat Equation
# 
# Integrate the heat equation for several days using a time step of 1 hour and a heat conductivity of  𝜈_𝑔 = 1.2e-6 [m2 s-1 ]. Plot the result. Once the code works, change the integration time. What happens if you integrate over a very long time?

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


def heat_equation(bc_surface, bc_bottom, depth, Nz, integration, dt):
    ''' Solves the heat equation
    bc_surface :: boundary condition at the surface
    bc_bottom  :: boundary condition at the bottom
    depth      :: depth of the domain [m]
    Nz         :: number of grid points
    integration:: number of iterations
    dt         :: time step [s]
    '''

    # Definitions
    dz    = depth/Nz  # Distance between grid points
    alpha = 1.2e-6    # Conductivity

    # Initialize temperature and depth field
    T = np.zeros(Nz)

    T[0] = bc_surface  # Set pen-ultima array to bc value (because the last grid cell
                          # is required to calculate the second order derivative)
    T[Nz-1] = bc_bottom      # Set the first elemnt to the bottom value

    # Create the solution vector for new timestep (we need to store the temperature values
    # at the old time step)    
    Tnew = T.copy()

    # Loop over all times
    for t in range(integration):
        pass
        # Loop over all grid points
        # ADD USER CODE HERE

        # Update old temperature array
        # ADD USER CODE HERE

        # Neumann boundary condition
        # ADD USER CODE HERE

    # return vertical temperature profile and grid spacing
    return T, dz




# In[2]:


# Plot results
fig, ax = plt.subplots(2,2,figsize=(12,12))

Nz = 100
T, dz = heat_equation(20, 0, 5, Nz, 24, 3600)
ax[0,0].plot(T,-dz*np.arange(Nz));
ax[0,0].set_xlabel('Temperature [ºC]')
ax[0,0].set_ylabel('Depth [m]')

T, dz = heat_equation(20, 0, 5, Nz, 24*14, 3600)
ax[0,1].plot(T,-dz*np.arange(Nz));
ax[0,1].set_xlabel('Temperature [ºC]')
ax[0,1].set_ylabel('Depth [m]')

T, dz = heat_equation(20, 0, 5, Nz, 24*30, 3600)
ax[1,0].plot(T,-dz*np.arange(Nz));
ax[1,0].set_xlabel('Temperature [ºC]')
ax[1,0].set_ylabel('Depth [m]')

T, dz = heat_equation(20, 0, 5, Nz, 24*365, 3600)
ax[1,1].plot(T,-dz*np.arange(Nz));
ax[1,1].set_xlabel('Temperature [ºC]')
ax[1,1].set_ylabel('Depth [m]')

plt.show()


# ### Heat equation with index arrays

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import math


def heat_equation_indices(bc_surface, bc_bottom, depth, Nz, integration, dt):
    ''' Solves the heat equation using index arrays
    bc_surface :: boundary condition at the surface
    bc_bottom  :: boundary condition at the bottom
    depth      :: depth of the domain [m]
    Nz         :: number of grid points
    integration:: number of iterations
    dt         :: time step [s]
    '''

    # Definitions
    dz    = depth/Nz # Distance between grid points
    alpha = 1.2e-6   # Conductivity

    # Define index arrays 
    # ADD USER CODE HERE


    # Initialize temperature and depth field
    T = np.zeros(Nz)

    T[0] = bc_surface     # Set pen-ultima array to bc value (because the last grid cell
                          # is required to calculate the second order derivative)
    T[Nz-1] = bc_bottom   # Set the first elemnt to the bottom value

    # Create the solution vector for new timestep (we need to store the temperature values
    # at the old time step)    
    Tnew = T.copy()

    # Loop over all times
    for t in range(integration):
        pass
    
        # ADD USER CODE HERE
        # Update temperature

        # Update old temperature array
        # ADD USER CODE HERE
    
        # Neumann boundary condition
        # ADD USER CODE HERE

    # return vertical temperature profile and grid spacing
    return T, dz





# In[4]:



# Plot results
fig, ax = plt.subplots(2,2,figsize=(12,12))

Nz = 100
T, dz = heat_equation_indices(20, 0, 5, Nz, 24, 3600)
ax[0,0].plot(T,-dz*np.arange(Nz));
ax[0,0].set_xlabel('Temperature [ºC]')
ax[0,0].set_ylabel('Depth [m]')

T, dz = heat_equation_indices(20, 0, 5, Nz, 24*14, 3600)
ax[0,1].plot(T,-dz*np.arange(Nz));
ax[0,1].set_xlabel('Temperature [ºC]')
ax[0,1].set_ylabel('Depth [m]')

T, dz = heat_equation_indices(20, 0, 5, Nz, 24*30, 3600)
ax[1,0].plot(T,-dz*np.arange(Nz));
ax[1,0].set_xlabel('Temperature [ºC]')
ax[1,0].set_ylabel('Depth [m]')

T, dz = heat_equation_indices(20, 0, 5, Nz, 24*365, 3600)
ax[1,1].plot(T,-dz*np.arange(Nz));
ax[1,1].set_xlabel('Temperature [ºC]')
ax[1,1].set_ylabel('Depth [m]')

plt.show()


# ### Time-dependent heat equation
# 
# Using the previous code, solve the Heat Equation using a temporal varying surface boundary condition. Use the following discretization: I = [0; 20 m], N = 40 grid points, 𝜈_𝑔 = 1.2e-6 [m2 s-1 ], and a daily time step. Integrate the equation for several years, e.g. 5 years. Plot the result as a contour plot. Also plot temperature time series in several depths. Discuss the plot!
# 

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import math


def heat_equation_time(depth, Nz, years):
    """ This is an example of an time-dependent heat equation using a 
    sinus wave temperature signal at the surface. The heat equation is solved for a 
    pre-defined number of years over the domain depth using Nz grid points."""

    # Definitions and assignments
    integration = 365*years    # Integration time in days
    dz  = depth/Nz             # Distance between grid points
    dt  = 86400                # Time step in seconds (for each day)
    K   = 1.2e-6               # Conductivity
 
    # Define index arrays 
    k  = np.arange(1,Nz-1)  # all indices at location i
    kr  = np.arange(2,Nz)   # all indices at location i+1
    kl  = np.arange(0,Nz-2) # all indices at location i-1

    # Initial temperature field
    T = np.zeros(Nz)

    # Create array for new temperature values
    Tnew = T

    # 2D-Array containing the vertical profiles for all time steps (depth, time)
    T_all = np.zeros((Nz,integration))

    
    # Time loop
    for t in range(integration):
        pass
    
        # Set top BC - Dirlichet condition
        # ADD USER CODE HERE

        # Set lower BC - Neumann condition
        # ADD USER CODE HERE
        
        # Update temperature using indices arrays
        # ADD USER CODE HERE
        
        # Copy the new temperature als old timestep values (used for the 
        # next time loop step)
        # ADD USER CODE HERE

        # Write result into the final array
        # ADD USER CODE HERE


    # return temperature array, grid spacing, and number of integration steps
    return T_all, dz, integration



# In[6]:


# Solve the heat equation
T_all, dz, integration = heat_equation_time(20, 40, 5)

# Create 2D mesh grid
# First create the y-axis values
y = np.arange(-20,0,dz)
# then the x-axis values
x = np.arange(integration)
# use the arrays to create a 2D-mesh
X, Y = np.meshgrid(x, y)

# Plot results on the mesh
plt.figure(figsize=(12,5))
plt.contourf(X,Y,T_all[::-1],25,origin='lower');

# Axis labels
plt.xlabel('Days')
plt.ylabel('Depth [m]')
plt.colorbar();


# In[7]:


# Plot temperature in several depths
plt.figure(figsize=(12,5))
plt.plot(T_all[0,:]);
plt.plot(T_all[10,:]);
plt.plot(T_all[20,:]);

