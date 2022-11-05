import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys                                          #for path to external scripts
#sys.path.insert(0,'/storage/emulated/0/github/cbse-papers/CoordGeo')         #path to my scripts
sys.path.insert(0,'/home/shashi/Desktop/CoordGeo')

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *

#Input parameters
A=np.array((1,1))   #fixed point through which circle passes A(p,q)
B=np.array((3,1))   #other end of  diameter through point A
D=np.array((2,0))   #the point at which circle touches x-axis 

d=LA.norm(A-B)     		 #diameter of the circle
r=d/2                      		 #radius of the circle
C=np.array((A+B)/2)            #centre of the circle

#Locus
V = np.array([[1,0],[0,0]])
u = np.array(([-1,-2]))
f =1

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
a = -2*eta/lamda[1]   

#Generating all shapes
p_x = parab_gen(p_y,a)
p_std = np.vstack((p_x,p_y)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
plt.plot(p[0,:], p[1,:],'r--',label='Locus of point B')


#Generating the circle
x_circ = circ_gen(C,r)

#Plotting the first circle
plt.plot(x_circ[0,:],x_circ[1,:],color='blue')

#Generating lines
x_AB=line_gen(A,B)
x_CD=line_gen(C,D)
plt.plot(x_CD[0,:],x_CD[1,:],'g--',label='radius')
plt.plot(x_AB[0,:],x_AB[1,:],color='red',label='diameter')

tri_coords = np.vstack((A,B,C,D)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A(p,q)','B(x,y)','C(h,k)','D']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
                 
ax=plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
				
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='best')
plt.axhline(y=0,color='black')
plt.axvline(x=0,color='black')
plt.axis('equal')
plt.grid()
plt.axis('equal')
plt.show()
