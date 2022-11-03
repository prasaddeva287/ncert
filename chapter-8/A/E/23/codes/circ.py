#Circle Assignment
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/namrath/Downloads/fwc/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#Input parameters
A = np.array(([0,3]))   #Given diameter end point of Circle 2
B = np.array(([-2,-1])) #Given diameter end point of Circle 2
L = np.array(([1,-1]))  #Point where line 2x+3y+1=0 touches Circle U2
m = np.array(([-3,2]))  #Line normal to tangent that touches Circle U2
K = np.array(([2,-1.67]))
J = np.array(([-4,2.33]))

#Centre and radius
U1 = -((A+B)/2) #Centre of circle U1 
u = (A+B)/2 
r1 = (LA.norm(A-B))/2 #Radius of Circle r1

#Computation
W = np.array(([2*((L.T)+(U1.T)),m.T]))
w = np.linalg.inv(W)
V = np.array(([(LA.norm(U1)**2 - LA.norm(L)**2 - (r1*r1)),-((m.T)@L)]))
h = w@V # w.U1 = V
U2 = -h #Centre of circle U2
r2 = LA.norm(U2-L) #Radius r2 of Circle U2

##Generating all linesS
x_KJ = line_gen(K,J) #Tangent ot Circle U2
x_R1 = line_gen(u,A) #Radius of Circle U1
x_R2 = line_gen(U2,L) #Radius of Circle U2

##Generating the circle
x_circ1= circ_gen(u,r1) #Circle U1
x_circ2= circ_gen(U2,r2) #Circle U2

# use set_position
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')

#Plotting all lines
plt.plot(x_KJ[0,:],x_KJ[1,:],label='$Tangent$')
plt.plot(x_R1[0,:],x_R1[1,:],label='$r1$')
plt.plot(x_R2[0,:],x_R2[1,:],label='$r2$')

#Plotting the circle
plt.plot(x_circ1[0,:],x_circ1[1,:],label='$Circle1$')
plt.plot(x_circ2[0,:],x_circ2[1,:],label='$Circle2$')

#Labeling the coordinates
tri_coords = np.vstack((u,U2,L)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['U1','U2','L']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

#lt.xlabel('$x$')
#plt.ylabel('$y$')
plt.legend()
plt.grid(True) # minor
plt.axis('equal')
