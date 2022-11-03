

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/student/CoordGeo')         #path to my scripts
#Circle Assignment
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA



#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#Input parameters
L = np.array(([3,4]))  #the point(a,b) through which circle passes
xp=np.arange(-5,15,1) #set of x points to draw locus

#Centre and radius of circle1
u = np.array([0,0])
r1 = np.array([5])

# plotting locus
yp_1=(L[0]**2+L[1]**2+r1**2)
yp_2=2*L[0]
yp_3=2*L[1]
yp=(yp_1-yp_2*xp)/yp_3
#yp=(50-6*xp)/8   #set of y points to draw locus
plt.plot(xp,yp,'r--',label='locus of center');

#Computation of center and radius of circle2
x=5	     #x point of center of circle2
y=(yp_1-yp_2*x)/yp_3
#y=np.array((50-6*x)/8)
U2 = np.array([x,y])
r2 = LA.norm(U2-L)

##Generating all linesS
x_R1 = line_gen(u,L) #Radius of Circle U1
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
plt.plot(x_R1[0,:],x_R1[1,:],label='$r1$')
plt.plot(x_R2[0,:],x_R2[1,:],label='$r2$')

#Plotting the circle
plt.plot(x_circ1[0,:],x_circ1[1,:],label='$Circle1$')
plt.plot(x_circ2[0,:],x_circ2[1,:],label='$Circle2$')

#Labeling the coordinates
tri_coords = np.vstack((u,U2,L)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['U1(0,0)','U2(x,y)','L(a,b)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.legend()
plt.grid(True) # minor
plt.savefig('cir.pdf')
plt.axis('equal')
plt.show()
