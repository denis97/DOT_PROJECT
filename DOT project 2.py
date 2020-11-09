#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'widget')
#%matplotlib inline
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import ticker
from matplotlib import cm
import numpy as np
import time
import plotly.graph_objects as go


# In[55]:



## Problem data

n=1 #refraction index
c= 30/n # cm/ns
mus= 10 #cm^-1
mua=0.1 # cm^-1
Dmua=0.1 #cm^-1
D=1/(3*mus) #cm
mm= 1e-1 # cm

dt=1
t=np.arange(0,8,dt) # ns

V=1 #cm^3 perturbation dimension

RP= (V*3/(4*np.pi))**(1/3) #cm perturbation radius

step=2.5

x=y=z= np.arange(0,64.1,step) * mm

xs=ys=np.arange(4,60.1,step) * mm

Xs, Ysd = np.meshgrid(xs,ys,xs,ys) #source-detector grid

Xsf=Xsd.flatten()
Ysf= Ysd.flatten()

X,Y,Z= np.meshgrid(x,y,z)  #space voxels

Xf,Yf,Zf=X.flatten(), Y.flatten(), Z.flatten()

xp,yp,zp = [1.5, 2.5 ,3]


# ### Absorption perturbation vector $\vec A$
# 
# Calculate A

# In[56]:


mask =(X-xp)**2+(Y-yp)**2+(Z-zp)**2 <RP**2

maskf =mask.flatten()

A= maskf*Dmua


# In[57]:


print("n° elements of A:",A.size)
A3D=A.reshape((x.size,y.size,z.size))
print("dimension of A reshaped:",A3D.shape)
print("Unit of measurment of A is cm^-1")


# In[58]:


plt.rcParams['figure.dpi'] = 200
plt.rcParams["figure.figsize"] = [5,5]

fig = plt.figure()
ax = fig.gca(projection='3d')

idx= A!=0

p=ax.scatter(Xf[idx], Yf[idx], Zf[idx], c=A[idx],cmap=cm.viridis,s=60, marker='o')
plt.xlim(0, 6.4)
plt.ylim(0, 6.4)
ax.set_zlim(0,6.4)
plt.colorbar(p)  
plt.show()


# In[82]:


graph= 1 # 1 o 2
if graph == 1:
    fffig = go.Figure(data=go.Volume(x=Xf, y=Yf, z=Zf,value=A,
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

# In[60]:


start = time.time()
i, m = np.ogrid[:Xsf.size,:Xf.size]
start = time.time()
rho= np.sqrt((Xsf[i]-Xf[m])**2 + (Ysf[i]-Yf[m])**2 +(Zf[m])**2)



W=0
tt=(t[1:] + t[:-1]) / 2
np.seterr(divide='ignore')
Sp=-1/(4*np.pi*D)*dt*((step*0.1)**3)*(np.divide(2,rho))

W= list(map(lambda t: Sp*np.exp(-(2*rho)**2 /(4*c*D*t)),tt))
W=np.sum(W,axis=0)/t[-1]

W[np.isinf(W)]=0
#W=np.nan_to_num(W)
end = time.time()
print(end - start)


# In[61]:


print("n° elements of W:",W.size)
print("dimension of W:",W.shape)
print("Unit of measurment of A is cm^-1")


# In[62]:


plt.figure()
plt.imshow(W,cmap=cm.jet,aspect='auto')
plt.colorbar()
plt.show()


# In[ ]:





# In[63]:


V_vox=np.sum(maskf)*(step*0.1)**3
print(V_vox)


# ### Measurments vector M

# In[64]:


Mf=np.inner(W,A)


# In[65]:


M=Mf.reshape((xsp.size,ysp.size))


# In[66]:


plt.figure()
plt.imshow(M,cmap='jet',aspect='equal',extent=[4,60,60,4])
plt.colorbar()
plt.show()


# ### Singular values & vectors

# In[76]:


[U,s,Vh]=np.linalg.svd(W)


# In[ ]:


plt.figure()
plt.imshow(np.diag(s),cmap='viridis',aspect='equal')
plt.colorbar()
plt.show()


# In[ ]:


plt.figure()
plt.imshow(U,cmap='viridis',aspect='equal')
plt.colorbar()
plt.show()


# In[ ]:


plt.close()
plt.figure()
Vmin=Vh.transpose()[:(min(W.shape)),:(min(W.shape))]
plt.imshow(Vmin,cmap='viridis',aspect='equal')
plt.colorbar()
plt.show()


# ### Inverse problem

# In[ ]:


S=np.diag(s)
S.shape
Sinv= np.zeros(W.shape).transpose()


# In[ ]:


i=range(s.size)
Sinv[i,i]=1/s[i]


# In[ ]:


Winv= np.linalg.multi_dot([Vh.transpose(),Sinv,U.transpose()])
Ap= Winv.dot(Mf)


# In[ ]:


App=np.linalg.pinv(W).dot(Mf)


# In[ ]:


fffig = go.Figure(data=go.Volume(x=Xf, y=Yf, z=Zf,value=Ap,
        opacity=0.1,# needs to be small to see through all surfaces
        surface_count=50 #needs to be a large number for good volume rendering
        ))
fffig.show()


# In[69]:


plt.figure()
plt.imshow(rho,cmap='jet',aspect='equal')
plt.colorbar()
plt.show()


# In[75]:


np.linalg.cond(W)

