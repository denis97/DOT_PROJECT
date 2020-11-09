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


# In[26]:



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

step=5

x=y=z= np.arange(0,64.1,step) * mm

xs=ys=np.arange(4,60.1,step) * mm

Xs, Ys,Xd,Yd = np.meshgrid(xs,ys,xs,ys) #source-detector grid

Xsf=Xs.flatten()
Ysf=Ys.flatten()
Xdf=Xd.flatten()
Ydf=Yd.flatten()

X,Y,Z= np.meshgrid(x,y,z)  #space voxels

Xf,Yf,Zf=X.flatten(), Y.flatten(), Z.flatten()

xp,yp,zp = [1.5, 2.5 ,1]


# ### Absorption perturbation vector $\vec A$
# 
# Calculate A

# In[27]:


mask =(X-xp)**2+(Y-yp)**2+(Z-zp)**2 <RP**2

maskf =mask.flatten()

A= maskf*Dmua


# In[28]:


print("n° elements of A:",A.size)
A3D=A.reshape((x.size,y.size,z.size))
print("dimension of A reshaped:",A3D.shape)
print("Unit of measurment of A is cm^-1")


# In[123]:


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


# In[30]:


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


# ### Sensitivity Matrix $W$

# In[47]:


start = time.time()
i,m= np.ogrid[:Xsf.size,:Xf.size]
start = time.time()

rho12= np.sqrt((Xsf[i]-Xf[m])**2 + (Ysf[i]-Yf[m])**2 +(Zf[m])**2) # distance source-pert
rho23= np.sqrt((Xdf[i]-Xf[m])**2 + (Ydf[i]-Yf[m])**2 +(Zf[m])**2) # distance detector-pert
r=np.sqrt((Xsf[i]-Xdf[i])**2 + (Ysf[i]-Ydf[i])**2)


W=0
tt=(t[1:] + t[:-1]) / 2
np.seterr(divide='ignore')
Sp=-1/(4*np.pi*D)*dt*((step*0.1)**3)*(1/rho12 + 1/rho23)

W= list(map(lambda t: Sp*np.exp((-(rho12+rho23)**2 +r**2) /(4*c*D*t)),tt))
W=np.sum(W,axis=0)/t[-1]

W[np.isinf(W)]=0
#W=np.nan_to_num(W)
end = time.time()
print(end - start)


# In[48]:


print("n° elements of W:",W.size)
print("dimension of W:",W.shape)
print("Unit of measurment of A is cm^-1")


# In[49]:


plt.figure()
plt.imshow(W,cmap=cm.jet,aspect='auto')
plt.colorbar()
plt.show()


# In[ ]:





# In[50]:


V_vox=np.sum(maskf)*(step*0.1)**3
print(V_vox)


# ### Measurments vector M

# In[51]:


Mf=np.inner(W,A)


# In[52]:


M=Mf.reshape((xs.size,ys.size,xs.size,ys.size))


# In[125]:


Msum=np.sum(M,(2,3))


# In[126]:


plt.close()
plt.figure()
plt.imshow(Msum,cmap='jet',aspect='equal',extent=[4,60,60,4])
plt.colorbar()
plt.show()


# In[56]:


M[np.where(M==M.min())]


# ### Singular values & vectors

# In[58]:


[U,s,Vh]=np.linalg.svd(W)


# In[59]:


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

# In[60]:


S=np.diag(s)
S.shape
Sinv= np.zeros(W.shape).transpose()


# In[61]:


i=range(s.size)
Sinv[i,i]=1/s[i]


# In[62]:


Winv= np.linalg.multi_dot([Vh.transpose(),Sinv,U.transpose()])
Ap= Winv.dot(Mf)


# In[63]:


fffig = go.Figure(data=go.Volume(x=Xf, y=Yf, z=Zf,value=Ap,
        opacity=0.1,# needs to be small to see through all surfaces
        surface_count=50 #needs to be a large number for good volume rendering
        ))
fffig.show()


# In[65]:


Ap.max()


# In[ ]:


App=np.linalg.pinv(W).dot(Mf)


# In[ ]:


plt.figure()
plt.imshow(rho,cmap='jet',aspect='equal')
plt.colorbar()
plt.show()


# In[ ]:


np.linalg.cond(W)


# In[109]:


plt.close()

