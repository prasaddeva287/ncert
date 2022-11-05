import numpy as np
import mpmath as mp
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
from pylab import *
from sympy import *

import sys                                         
sys.path.insert(0,"/home/shashi/Desktop/CoordGeo")   
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *


#if using termux
#import subprocess
#import shlex
#end if

#for parabola
V = np.array([[1,-1],[-1,1]])
u = np.array(([-4,-4]))
f =16

Vertex= np.array(([1,1]))
Focus = np.array(([2,2])) #h

def affine_transform(P,c,x):
    return P@x + c

#Transformation 
lamda,P = LA.eigh(V)
if(lamda[1] == 0):  # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda)
    P = np.flip(P,axis=1)

eta = u@P[:,0]
a = np.vstack((u.T + eta*P[:,0].T, V))
b = np.hstack((-f, eta*P[:,0]-u)) 
center = LA.lstsq(a,b,rcond=None)[0]
O = center 
n = np.sqrt(lamda[1])*P[:,0]
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = np.array(([0,0.5]))
fl = LA.norm(F)

#pmeters to generate parabola
num_points = 8000
delta = 100*np.abs(fl)/10
p_y = np.linspace(-4*np.abs(fl)-delta,4*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y

#Generating all shapes
p_x = parab_gen(p_y,a)
p_std = np.vstack((p_x,p_y)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
plt.plot(p[0,:], p[1,:],label='parabola')

#Labeling the coordinates
tri_coords = np.vstack((Vertex,Focus)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['V(1,1)','F(2,2)']
for i, txt in enumerate(vert_labels): 
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
#if using termux
#plt.savefig(os.path.join(script_dir, fig_relative))
#subprocess.run(shlex.split("termux-open "+os.path.join(script_dir, fig_relative)))
#else
#plt.legend()
x=np.linspace(-8,8,10)
y=-x
y1=x
plt.plot(x,y1,'r--',label='axis of symmetry')
plt.plot(x,y,'red',label='directrix')
plt.legend(loc='upper left')
plt.axhline(y=0,color='black')
plt.axvline(x=0,color='black')
plt.axis('equal')
plt.grid()
plt.show()
