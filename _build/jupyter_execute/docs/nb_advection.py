#!/usr/bin/env python
# coding: utf-8

# (advection:exercises)=
# ### Advection diffusion equation
# 
# **Task 4**: Solve the Advection-Diffusion equation, with the following initial and boundary conditions: at t=0 , 𝑐$_0$=0; for all subsequent times, 𝑐=0 at x=0, 𝑐=1 at 𝑥=𝐿=1, 𝑢=1.0 and K=0.1. Integrate over 0.05 s with a Δ𝑡=0.0028. Plot the results and the dimensionless time scales. Increase gradually Δ𝑡 and analyse the results. Once you understand what is happening, set again Δ𝑡=0.0028 and gradually increase the wind speed. Discuss the results.
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math

get_ipython().run_line_magic('matplotlib', 'inline')


def advection_diffusion(u, K, integration, dt, Nx):
    """ Simple advection-diffusion equation.
    
    integration :: Integration time in seconds
    Nx          :: Number of grid points
    dt          :: time step in seconds
    K           :: turbulent diffusivity
    u           :: Speed of fluid
    """
    
    # Definitions and assignments
    a   = 0.                # Left border
    b   = 1.                # Right border
    dx  = (b-a)/Nx          # Distance between grid points

    # Define the boundary conditions
    bc_l  = 1       # Left BC
    bc_r  = 0       # Right BC

    # Define index arrays 
    k   = np.arange(1,Nx-1)
    kr  = np.arange(2,Nx)
    kl  = np.arange(0,Nx-2)

    # Initial quantity field
    phi = np.zeros(Nx)
                 
    # Set boundary condiiton
    phi[Nx-1] = bc_r
    phi[0] = bc_l

    # Dimensionless parameters
    # CHANGE THE TWO LINES OF CODE HERE
    c = (u*dt)/dx
    d = (K*dt)/(dx**2)

    # Time loop
    t = 0

    while t <= integration:
        # Update flux
        phi[k] = (1-2*d) * phi[k] + (d-(c/2))*phi[kr] - (d+(c/2))*phi[kl]
        
        # Update time
        t = t + dt     
        
    return(phi, dx, u, K, c, d)



# In[3]:


phi, dx, u, K, c, d = advection_diffusion(u=1.0, K=0.1, integration=0.05, dt=0.0028, Nx=40)
     
print('Dimensionless parameter c: {:.4f}'.format(c))
print('Dimensionless parameter d: {:.4f}'.format(d))

plt.figure(figsize=(12,5))
plt.plot(phi)
plt.show()


# In[ ]:


# Define the CFL criteria
CFL = 0.7
print("required dt (advection) <= {:.4f} s".format((CFL * dx)/u))
print("required dt (diffusion) <= {:.4f} s".format(((CFL * dx**2))/(2*K)))
print('')   


# **Task 5**: Solve the Advection-Diffusion equation, with the following initial impulse signal and boundary conditions: 
# 
# \begin{align}
# c(n,0) &=exp^{\left(-\left(\frac{(x-10)}{2}\right)^2\right)} \\
# c(0,t) &=0 \\
# c(L,t) &=\frac{\partial c}{\partial x}=0
# \end{align}
# 
# Integrate the equation with K=0.1, u=1.0 over 0.05 s with a Δ𝑡=0.0028. Plot the results and the dimensionless time scales. Increase gradually Δ𝑡 and plot and analyse the results for different integration times.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import math

get_ipython().run_line_magic('matplotlib', 'inline')


def advection_diffusion(u, K, integration, dt, Nx):
    """ Simple advection-diffusion equation.
    
    integration :: Integration time in seconds
    Nx          :: Number of grid points
    dt          :: time step in seconds
    K           :: turbulent diffusivity
    u           :: Speed of fluid
    """
    
    # Definitions and assignments
    a   = 0.                # Left border
    b   = 1.                # Right border
    dx  = (b-a)/Nx          # Distance between grid points

    # Define the boundary conditions
    bc_l  = 0       # Left BC
    bc_r  = 0       # Right BC

    # Define index arrays 
    k   = np.arange(1,Nx-1)
    kr  = np.arange(2,Nx)
    kl  = np.arange(0,Nx-2)

    # Initial temperature field
    phi = np.exp(-((np.arange(Nx)-10)/2)**2)
                 
    # Set boundary condiiton
    phi[Nx-1] = bc_r
    phi[0] = bc_l

    # Dimensionless parameters
    d = (K*dt)/(dx**2)
    c = (u*dt)/dx

    # Time loop
    t = 0
  
    while t <= integration:

        # Set BC (Neumann condition)
        phi[Nx-1] = phi[Nx-2]
        
        # Update flux
        phi[k] = (1-2*d)*phi[k] + (d-(c/2))*phi[kr] + (d+(c/2))*phi[kl]

        # Increate time
        t = t + dt  
        
    return(phi, dx, u, K, c, d)




# In[ ]:


phi, dx, u, K, c, d = advection_diffusion(u=1.0, K=0.1, integration=0.05, dt=0.0028, Nx=40)

print("required dt (advection) <= {:.4f} s".format((0.3*dx)/u))
print("required dt (diffusion) <= {:.4f} s".format(((dx**2))/(2*K)))
print('')       
print('Courant number (c): {:.4f} (C<1)'.format(c))
print('Characteristic diffusion time (d): {:.4f}'.format(d))

plt.figure(figsize=(12,5))
plt.plot(phi)
plt.show()


# In[ ]:


# Simulate evolution for different time steps
# Make plots
# ADD USER CODE HERE


# **Task 6**: Starting with the heat equation above simplify the equation to model the temperature evolution in the boundary layer from the surface up to H=2 km height. Assume a fair-weather condition with a subsidence of -0.001 m s-1. Also assume horizontal homogeneity. Parameterize the heat flux using the eddy-diffusivity closure with K=0.25 m s-2. Solve the simplified equation using the following initial and boundary conditions:
# 
# \begin{align}
# \theta(z,0)=290~K \\
# \overline{w'\theta'}(z,0)=0~W~m^{−2} \\
# \theta(0,t)=290+10 \cdot sin\left(\frac{2\pi \cdot t}{86400}\right)~K \\
# \theta(H,t)=\frac{\partial \theta}{\partial z}=0.01~K~m^{-1}
# \end{align}
# 
# - What happens when you increase the subsidence to -0.01 m s$^{-1}$? 
# - Plot the kinematic heat flux.
# - What is the maximum heat flux in W m$^{-2}$? Is this a realistic values for a fair-weather condition?
# - Calculate the heat rate.

# In[5]:


def boundary_layer(w, K, integration, dt, Nz, H):
    """ Simple advection-diffusion equation.
    
    integration :: Integration time in seconds
    Nz          :: Number of grid points
    dt          :: time step in seconds
    K           :: turbulent diffusivity
    u           :: Speed of fluid
    """
    
    # Definitions and assignments
    a   = 0.                # Left border
    b   = H                 # Right border
    dz  = (b-a)/Nz          # Distance between grid points

    # Define the boundary conditions
    bc_t  = 0       # top BC
    bc_b  = 0       # bottom BC

    # Define index arrays 
    k   = np.arange(1,Nz-1)
    kr  = np.arange(2,Nz)
    kl  = np.arange(0,Nz-2)

    # Initial temperature field
    theta = 290 * np.ones(Nz)
    theta_all = np.zeros((Nz,int(integration/dt)))

    cov = np.zeros(Nz)
    cov_all = np.zeros((Nz,int(integration/dt)))
    
    # Dimensionless parameters
    d = (K*dt)/(dz**2)
    c = (w*dt)/dz
    
    t = 0
    
    for idx in range(int(integration/dt)):
        
        # Set BC top
        theta[Nz-1] = theta[Nz-2] + 0.01 * dz
        
        # Set BC surface
        theta[0] = 290 + 10.0 * np.sin((2*pi*t)/86400)
        
        # Update flux
        theta[k] = (1-2*d)*theta[k] + (d-(c/2))*theta[kr] + (d+(c/2))*theta[kl]
        theta_all[:,idx] = theta[:]
        
        # Calculate and store the covariance
        cov[k] = -K * ((theta[kr]-theta[kl]) / (2*dz))
        cov_all[:,idx] = cov[:]
             
        # Increase time step
        t = t + dt
        
    return(theta_all, cov_all, dz, u, K, c, d)


# In[ ]:


def make_plot(data, x, z, levels, title, unit, xlab, zlab, cmap='RdBu_r'):
    """ Useful function for plotting 2D-fields as contour plot"""
    
    # Create figure
    fig, ax = plt.subplots(1,1,figsize=(18,5));
    cn0 = ax.contourf(x,z,data,10,origin='lower',levels=levels,cmap=cmap);
    
    # Add the colorbar and set ticks and labels
    cbar= fig.colorbar(cn0, ax=ax, orientation='vertical')
    cbar.set_label(label=unit, size=16)
    cbar.ax.tick_params(labelsize=14)
    
    # Add labels and modify ticks
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(zlab, fontsize=14)
    ax.set_title(title)
    
    return ax


# In[ ]:


Nz = 200
H = 500
integration = 86400*3
dt = 1

phi1, cov, x, z, u, K, c, d = boundary_layer(w=-0.001, K=0.25, integration=integration, dt=dt, Nz=Nz, H=H)

# Create 2D mesh grid
ax = make_plot(phi1, x=x, z=z, levels=21, title='Theta', unit='K', xlab='Hours', zlab='Height', cmap='RdBu_r')

# Correct the ticks
ax.set_xticks(x[x%(3600*6)==0]);
ax.set_xticklabels(list(map(str,(x[x%(3600*6)==0]/3600))), size=10, weight='normal');


# In[ ]:


# Plot the heat fluxes

# Create 2D plot for the covariance
ax = make_plot(cov[:], x=x, z=z, levels=21, title='Theta', unit='W m$^{-2}$', xlab='Hours', zlab='Height', cmap='RdBu_r')
# Correct the ticks
ax.set_xticks(x[x%(3600*6)==0]);
ax.set_xticklabels(list(map(str,(x[x%(3600*6)==0]/3600))), size=10, weight='normal');

# Create 2D plot for the kinematic heat flux
ax = make_plot(cov[:]*1004, x=x, z=z, levels=21, title='Kinematic heat flux', unit='K m$^{-1}$', xlab='Hours', zlab='Height', cmap='RdBu_r')
# Correct the ticks
ax.set_xticks(x[x%(3600*6)==0]);
ax.set_xticklabels(list(map(str,(x[x%(3600*6)==0]/3600))), size=10, weight='normal');


# **Task 7**: Intense boundary layer convection may develop when cold air masses are advected over relatively warm surfaces. Develop a simple model for this by assuming that the time evolution of the boundary layer is determined by the vertical turbulent heat transport and the horizontal heat advection. Make the following assumptions: [Hint: use the eddy-diffusivity closure and the upwind scheme for the advection flux]

# ![Screenshot%202022-06-18%20at%2018.27.03.png](attachment:Screenshot%202022-06-18%20at%2018.27.03.png)

# In[6]:


import random

def boundary_layer_evolution(u, K, dx, dz, Nx, Nz, hours, dt):
    """ Simple advection-diffusion equation.
    
    integration :: Integration time in seconds
    Nz          :: Number of grid points
    dt          :: time step in seconds
    K           :: turbulent diffusivity
    u           :: Speed of fluid
    """
       
    # Some definitions
    integration = hours*3600
    
    # Define index arrays 
    k   = np.arange(1,Nx-1)
    kr  = np.arange(2,Nx)
    kl  = np.arange(0,Nx-2)
    
    m   = np.arange(1,Nz-1)
    mu  = np.arange(2,Nz)
    md  = np.arange(0,Nz-2)

    # Initial temperature field
    theta = 273 * np.ones((Nz,Nx))
    cov =   np.zeros((Nz,Nx))
    adv =   np.zeros((Nz,Nx))
    
    # Define the boundary conditions
    # Set BC surface
    theta[0, :] = 273
    
    # Initialize lake boudary
    lake_from = 50
    lake_to = 150
    theta[0, lake_from:lake_to] = 283
    
    # Dimensionless parameters
    c = (u*dt)/dx
    d = (K*dt)/(dz**2)

    for idx in range(int(integration/dt)):

        # Set BC top
        theta[Nz-1, :] = theta[Nz-2, :] + 0.01 * dz
        
        # Set BC right
        theta[:,Nx-1] = theta[:,Nx-2]
        
        # Keep track of old temperature data
        old = theta
            
        # First update grid cells in x direction (loop over all z values)
        for x in range(1,Nx-1):
            theta[m,x] = theta[m,x] + ((K*dt)/(dz**2))*
                                    (old[mu,x]+old[md,x]-2*old[m,x])

        # Then update grid cells in z direction (loop over all x values)
        for z in range(1,Nz-1):
            theta[z,k] = theta[z,k] - ((u*dt)/(dx))*(old[z,k]-old[z,kl])

        
    return theta, c, d, np.arange(0, Nx*dx, dx), np.arange(0, Nz*dz, dz)



# ## theta, cov, adv, c, d, x, z = boundary_layer_evolution(u=5, K=0.02, dx=500, dz=5, Nx=250, Nz=20, hours=10, dt=75)

# In[ ]:


# Create 2D plot for the covariance
ax = make_plot(theta, x=x/1000, z=z, levels=21, title='Heat flux', unit='W m$^{-2}$', 
               xlab='Distance', zlab='Height', cmap='RdBu_r')


# In[ ]:


# Create 2D plot for the covariance
ax = make_plot(cov*1004, x=x/1000, z=z, levels=21, title='Kinematic heat flux', unit='K m s$^{-1}$', 
               xlab='Distance', zlab='Height', cmap='RdBu_r')


# In[ ]:


# Create 2D plot for the covariance
ax = make_plot((cov[:,:]/5)*3600, x=x/1000, z=z, levels=21, title='Warming rate per hour due to turbulent mixing', 
               unit='K h$^{-1}$', 
               xlab='Distance', zlab='Height', cmap='RdBu_r')


# In[ ]:


# Create 2D plot for the covariance
ax = make_plot(adv*3600, x=x/1000, z=z, levels=21, title='Warming rate per hour due to advection', 
               unit='K h$^{-1}$', 
               xlab='Distance', zlab='Height', cmap='RdBu_r')

