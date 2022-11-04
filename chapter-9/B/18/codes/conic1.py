import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys, os                                 
sys.path.insert(0,'/sdcard/matrices/circle/CoordGeo')  #path to my scripts
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import parab_gen

#if using termux
import subprocess
import shlex
#end if

def affine_transform(P,c,x):
    return P@x + c
    
#Input parameters
V = np.array([[0,0],[0,1]])
u = np.array(([-2,0]))
o=np.array(([0,0]))
f = 0


#for plotting parabola
lamda,P = LA.eigh(V) 
if(lamda[1] == 0):      # If eigen value negative, present at start of lamda
    lamda = np.flip(lamda)       
    P = np.flip(P,axis=1)
eta = u@P[:,0] 
a = np.vstack((u.T + eta*P[:,0].T, V))     
b = np.hstack((-f, eta*P[:,0]-u))
center = LA.lstsq(a,b,rcond=None)[0]
O=center
n = np.sqrt(lamda[1])*P[:,0]
print(n)
#n = np.array(([0,1]))
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)  
print(c)    
F = (c*n - u)/lamda[1]
fl = LA.norm(F)   
x=12

num_points =50
delta = 2*np.abs(fl)/10
p_y = np.linspace(-10*np.abs(fl)-delta,10*np.abs(fl)+delta,num_points)
g = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y

#normal vectors of perpendicular tangents
h=np.array((-1/4,5))
e1=np.array((1,0))
R=np.array((0,-1,1,0)).reshape(2,2)
m=R@h
qq=e1/(e1.T@h)
w1=m.T@V@m
w2=V@qq+u
w3=qq.T@V@qq+2*u.T@qq+f
w4=-m.T@w2
w5=(m.T@w2)**2-w3*w1
ww1=np.sqrt(w5)
w6_1=(w4+ww1)/w1
w6_2=(w4-ww1)/w1

tt1=e1/(e1.T@h)
nn_1=tt1+w6_1*R@h
nn_2=tt1+w6_2*R@h

print(nn_1,nn_2)



#Generating the Locus of P
m=omat@nn_2
k1=-0.002
k2=0.008
x_BC = line_dir_pt(m,h,k1,k2)


##Generating all shapes
p_x = parab_gen(p_y,g)
p_std = np.vstack((p_x,p_y)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T



#Plotting all shapes

plt.plot(p[0,:], p[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])

#Labeling the coordinates

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/matrices/conics/CoordGeo/circle1.pdf')
subprocess.run(shlex.split("termux-open /sdcard/matrices/conics/CoordGeo/circle1.pdf"))
plt.show()

