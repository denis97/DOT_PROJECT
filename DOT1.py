# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:04:09 2020

@author: dilia
"""

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

dim=21

## Problem data
n=1 #refraction index
c= 30/n # cm/ns
mus= 10 #cm^-1
mua=0.1 # cm^-1
Dmua=0.1 #cm^-1
D=1/(3*mus) #cm

t=np.linspace(0,3,1000)[1:] # ns

mus= 10 *cm**-1
mua=0.1  *cm**-1
Dmua=0.1 *cm**-1
D=1/(3*mus)
ns=1e-9

t=np.linspace(0,3,10000)[1:] * ns

V=1 *cm**3 #perturbation dimension

RP= (V*3/4)**(1/3) #cm perturbation radius


rs=np.array([0,0,0]) *cm
rd= np.array([0,0,0])*cm
rp=np.array([0, 0 ,2])* cm

r= np.linalg.norm(rd)
Phi0= (c*((4*np.pi*c*D*t)**(-3/2)))*np.exp(-c*mua*t)

def perturbation(rs,rd,rp):
    
    r= np.linalg.norm(rd)
    xs,ys,zs= rs
    xd,yd,zd= rd
    xp,yp,zp = rp
    
    X,Y,Z= np.mgrid[-6:6:dim*1j,-6:6:dim*1j,-6:6:dim*1j]
    
    def midpoints(x):
        sl = ()
        for i in range(x.ndim):
            x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
            sl += np.index_exp[:]
        return x
    
    xm,ym,zm= midpoints(X), midpoints(Y),midpoints(Z)
    
    mask = (xm-xp)**2+(ym-yp)**2+(zm-zp)**2 <RP**2
    dmua=np.zeros(mask.shape)
    dmua[mask]=Dmua #cm^-1
    

    h=X[1,0,0]-X[0,0,0]
    
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.voxels(X, Y, Z, mask,edgecolor='k')
    
    # plt.show()
    
    
    rho12= np.sqrt((xm-xs)**2+(ym-ys)**2+(zm-zs)**2)
    rho23= np.sqrt((xm-xd)**2+(ym-yd)**2+(zm-zd)**2)
    
    def sumInt(tt):
           return [(h**3)*np.sum(dmua*(1/rho12 + 1/rho23) * np.exp(- (rho12 +rho23)**2 /(4*c*D*t))) 
                   for t in tt]
    
    delPhi0= -(c**2/(4*np.pi*c*D)**(5/2))*(t**-3/2)*np.exp(-c*mua*t) * sumInt(t)
    return delPhi0

delPhi0= perturbation(rs,rd,rp)   
Contrast= delPhi0/Phi0


fig2= plt.figure(figsize=(16, 9))
ax=[]

ax.append(fig2.add_subplot(311))
ax[0].plot(t,Phi0)
ax[0].set(ylabel="$ \Phi_0 $", yscale="linear")


ax.append(fig2.add_subplot(312))
ax[1].plot(t,delPhi0)
ax[1].set(ylabel="$ \delta \Phi_0 $")

ax.append(fig2.add_subplot(313))
ax[2].plot(t,Contrast)
ax[2].set(ylabel="$ Contrast$" )
ax[2].autoscale()


fig2.show()