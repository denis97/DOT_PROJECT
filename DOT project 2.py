#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib widget
#%matplotlib inline
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import ticker
from matplotlib import cm
import numpy as np
import time
import plotly.graph_objects as go


# In[2]:



## Problem data

n=1 #refraction index
c= 30/n # cm/ns
mus= 10 #cm^-1
mua=0.1 # cm^-1
Dmua=0.1 #cm^-1
D=1/(3*mus) #cm
mm= 1e-1 # cm

dt=0.5
t=np.arange(0,8,dt) # ns

V=1 #cm^3 perturbation dimension

RP= (V*3/(4*np.pi))**(1/3) #cm perturbation radius

step=2.5

x=y=z= np.arange(0,64.1,step) * mm

xsp=ysp=np.arange(4,60.1,step) * mm

Xsd, Ysd = np.meshgrid(xsp,ysp)

Xsf=Xsd.flatten()
Ysf= Ysd.flatten()

X,Y,Z= np.meshgrid(x,y,z)

Xf,Yf,Zf=X.flatten(), Y.flatten(), Z.flatten()

rs=np.array([0,0,0])
rd= np.array([0,0,0])
xp,yp,zp = [1.5, 2.5 ,2.0]

r= np.linalg.norm(rd)
#Phi0= 1e13*(c*((4*np.pi*c*D*t)**(-3/2)))*np.exp(-c*mua*t)


# ### Absorption perturbation vector $\vec A$
# 
# Calculate A

# In[3]:


mask =(X-xp)**2+(Y-yp)**2+(Z-zp)**2 <RP**2

maskf =mask.flatten()

A= maskf*Dmua


# In[4]:


print("n° elements of A:",A.size)
A3D=A.reshape((x.size,y.size,z.size))
print("dimension of A reshaped:",A3D.shape)
print("Unit of measurment of A is cm^-1")


# In[5]:


plt.rcParams['figure.dpi'] = 600
plt.rcParams["figure.figsize"] = [10,7]

fig = plt.figure()
ax = fig.gca(projection='3d')

idx= A!=0

ax.scatter(Xf[idx], Yf[idx], Zf[idx], c=A[idx],cmap=cm.viridis,s=60, marker='o')
plt.xlim(0, 6.4)
plt.ylim(0, 6.4)
ax.set_zlim(0,6.4)
    
plt.show()


# In[6]:


graph= 2 # 1 o 2
if graph == 1:
    fffig = go.Figure(data=go.Volume(x=Xf, y=Yf, z=Zf,value=A,
        isomin=0.01,
        isomax=0.1,
        opacity=0.1,# needs to be small to see through all surfaces
        surface_count=50 #needs to be a large number for good volume rendering
        ))
    fffig.show()

if graph ==2:
    
    fig = plt.figure()
    xi,yi,zi = np.indices(np.array(A3D.shape)+1)/A3D.shape[1] *x[-1]
    ax = fig.gca(projection='3d')
    ax.voxels(xi,yi,zi,A3D,edgecolor='k')

    plt.show()


# In[ ]:





# ### Sensitivity Matrix $W$

# In[7]:


start = time.time()
i, m = np.ogrid[:Xsf.size,:Xf.size]
start = time.time()
rho= (Xsf[i]-Xf[m])**2 + (Ysf[i]-Yf[m])**2 +(Zf[m])**2



W=0
tt=(t[1:] + t[:-1]) / 2
np.seterr(divide='ignore')
Sp=-1/(4*np.pi*D)*dt*((step*0.1)**3)*(np.divide(2,rho))

W= list(map(lambda t: Sp*np.exp(-(2*rho)**2 /(4*c*D*t)),tt))
W=np.sum(W,axis=0)/t[-1]


W=np.nan_to_num(W)
end = time.time()
print(end - start)


# In[8]:


print("n° elements of W:",W.size)
print("dimension of W:",W.shape)
print("Unit of measurment of A is cm^-1")


# In[9]:


plt.figure()
plt.imshow(W,cmap=cm.jet,aspect='auto')
plt.colorbar()
plt.show()


# In[ ]:





# In[10]:


V_vox=np.sum(maskf)*(step*0.1)**3
print(V_vox)


# In[11]:


Mf=np.inner(W,A)


# In[12]:


M=Mf.reshape((xsp.size,ysp.size))


# In[13]:


plt.figure()
plt.imshow(M,cmap='jet',aspect='equal',extent=[4,60,60,4])
plt.colorbar()
plt.show()


# In[14]:


W.size


# In[ ]:




