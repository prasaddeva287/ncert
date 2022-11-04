#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import math as m
import random as r
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sympy as sym
import math
import sympy
import sys
import sys, os                                          #for path to external scripts
script_dir = os.path.dirname(__file__)
lib_relative = '../../../CoordGeo'
fig_relative = '../figs/fig1.pdf'
sys.path.insert(0,'/sdcard/conic/CoordGeo')

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import parab_gen
from sympy import Poly,roots,simplify
from sympy import*
#if using termux
import subprocess
import shlex
#end if
x=sym.Symbol('x1')
y=sym.Symbol('y1')
X=np.array(([x,y]))

def affine_transform(P,c,x):
    return P@x + c



def parab_1(x):
    y=(x**2)-5*x+6
    return(y)

#Input parameters
A=np.array(([2,0]))
B=np.array(([3,0]))
V = np.array([[1,0],[0,0]])
u = np.array(([-5/2,-1/2]))
f = 6

tang1=(V@A+u).T@X+u.T@A+f
print(tang1)
n1=(V@A+u)
m1=omat@n1
tang2=(V@B+u).T@X+u.T@B+f
print(tang2)


n2=(V@B+u)

m2=omat@n2
#points on lines
k1=-20
k2=20
x_AB = line_dir_pt(m1,A,k1,k2)
x_CD = line_dir_pt(m2,B,k1,k2)
lamda,P = LA.eigh(V)
if(lamda[1] == 0):      # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda)
    P = np.flip(P,axis=1)
eta = u@P[:,0]
a = np.vstack((u.T + eta*P[:,0].T, V))
b = np.hstack((-f, eta*P[:,0]-u)) 
center = LA.lstsq(a,b,rcond=None)[0]
O = center 
n = np.sqrt(lamda[1])*P[:,0]
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = (c*n - u)/lamda[1]
fl = LA.norm(F)
m = omat@n
d = np.sqrt((m.T@(V@F + u))**2 - (F.T@V@F + 2*u.T@F + f)*(m.T@V@m))
k1 = (d - m.T@(V@F + u))/(m.T@V@m)
k2 = (-d - m.T@(V@F + u))/(m.T@V@m)
A = F + k1*m
B = F + k2*m



num_points = 50
delta = 2*np.abs(fl)/10
p_x1 = np.linspace(-5,10,100)
p_y1 = np.linspace(-10,10,100)
p_y = np.linspace(-2*np.abs(fl)-delta,2*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y'


##Generating all shape
p_x = parab_gen(p_y,a)
p_y1 = parab_1(p_x1)
p_std = np.vstack((p_x1,p_y1)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T

# Generating lines after transforming points



#Plotting all shapes
plt.plot(p_x1,p_y1)
#plt.xlim([-6,6])
#plt.ylim([10,0])
#Labeling the coordinates
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_CD[0,:],x_CD[1,:])#,label='$Diameter$')

#Labeling the coordinates
tri_coords = np.vstack((A,B)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
#plt.axis('equal')







##if using termux
plt.savefig('/sdcard/conic/fig.pdf')
subprocess.run(shlex.split("termux-open /sdcard/conic/fig.pdf"))
##else
plt.show()
