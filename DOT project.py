#!/usr/bin/env python
# coding: utf-8

# #  DIFFUSE OPTICAL MICROSCOPY PROJECT
# 
# ## Phase 1: Time domain  DOT
# 
# ### Objectives:
#  - Obtain Formula for absorption
#  - Study Contrast function
#  
# ### Formula on Contrast :
# 
# 1. write fluence  $\phi_0(\mu_a,\mu_s',t) $ for an homogeneous medium:
# 
# $$ \phi_0(\mu_a,\mu_s',t)= \frac{c}{(4\pi cD t)^{3/2}}\cdot exp(-c\mu_a t)$$
# 
# 
# 2. write fluence perturbation  $ \delta \phi_0(\mu_a,\mu_s',t, \delta \mu_a,V,\vec{r}) $ :
# 
# $$ \delta \phi_0(\mu_a,\mu_s',t, \delta \mu_a,V,\vec{r}) = -\frac{c^2}{(4\pi D c)^{5/2} t^{3/2}} \cdot exp(-\mu_a c t) \int_{V_i} \delta \mu_a (\vec {r_p}) \left(\frac{1}{\rho_{12}} + \frac{1}{\rho_{23}}\right) exp\left\{-\frac{\left(\rho_{12} +\rho_{23}\right)^2}{4cDt}\right\} d^3 \vec{r_p}$$
# 
# 
# 3. write $ C(t) \equiv  \delta \phi_0\big/ \phi_0  $
# 
# 
# ### Plots

# In[29]:





# In[1]:


get_ipython().run_line_magic('matplotlib', 'widget')

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


# In[97]:


c= 30 # cm/ns
mus= 10 #cm^-1
mua=0.1 # cm^-1
dim=21
D=1/(3*mus) #cm
pi= np.pi
t=np.linspace(0,3,10000) # ns
t=t[1:] # ns
V=1 #cm^3
rp= (V*3/4)**(1/3) #cm

xs,ys,zs=[0,0,0]
xd,yd,zd= [0,0,2]
r= np.linalg.norm([xd,yd,zd])
xp,yp,zp=[0, 0, 1]

X,Y,Z= np.mgrid[-6:6:dim*1j,-6:6:dim*1j,-6:6:dim*1j]

def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

xm,ym,zm= midpoints(X), midpoints(Y),midpoints(Z)

mask = (xm-xp)**2+(ym-yp)**2+(zm-zp)**2 <rp
dmua=np.zeros(mask.shape)
dmua[mask]=0.1 #cm^-1

Phi0= (c/(4*pi*c*D*t)**(3/2))*np.exp(-(r**2/(4*D*c*t))-c*mua*t)

h=X[1,0,0]-X[0,0,0]


# In[78]:


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(X, Y, Z, mask,edgecolor='k')

plt.show()


# In[98]:


rho12= (xm-xs)**2+(ym-ys)**2+(zm-zs)**2
rho23= (xm-xd)**2+(ym-yd)**2+(zm-zd)**2

def sumInt(tt):
       return [(h**3)*np.sum(dmua*(1/rho12 + 1/rho23) * np.exp(- (rho12 +rho23)**2 /(4*c*D*t))) 
               for t in tt]

delPhi0= -(c**2/(4*pi*c*D)**(5/2))*(t**2/3)*np.exp(-(r**2/(4*D*c*t))-c*mua*t) * sumInt(t)
#%%

Contrast= delPhi0/Phi0


# In[99]:



plt.close(fig2)

fig2= plt.figure(figsize=(12, 9))
ax=[]

ax.append(fig2.add_subplot(311))
ax[0].plot(t,Phi0)
ax[0].set(ylabel="$ \Phi_0 $")


ax.append(fig2.add_subplot(312))
ax[1].plot(t,delPhi0)
ax[1].set(ylabel="$ \delta \Phi_0 $")

ax.append(fig2.add_subplot(313))
ax[2].plot(t,Contrast)
ax[2].set(ylabel="$ Contrast$")


fig2.show()


# In[ ]:




