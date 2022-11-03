import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          
sys.path.insert(0,'/sdcard/gokul/matrices/circle/codes/CoordGeo')    

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *


import subprocess
import shlex


def perpendicular( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

I = np.eye(2)
V = I
u = np.array(([-1,-2]))
f = -20
A = -u 
r=np.sqrt(u@(u)-f)
B=np.array(([1,7]))
D=np.array(([4,-2]))


#parameters for line BC
m11=A-B
m1 =perpendicular(m11)
#n1 = omat@m1


#parameters for line CD
m22=A-D
m2 = perpendicular(m22)
#n2 = omat@m2

#Intersection of BD and AQ
#C = line_intersect(n1,B,n2,D)



#Generating the director vectors

matM = np.block([[m1],[m2]]).T
lam = LA.solve(matM,D-B)
C = B + lam[0]*m1
#print(C)

v1=A-B
v2=A-C
ar_t1=0.5*np.linalg.norm((np.cross(v1,v2)))
ar=2*ar_t1
print("Area of quadrilateral is ", ar)


#Generating all lines
xBA = line_gen(B,A)
xAD = line_gen(A,D)
xBC = line_gen(B,C)
xDC = line_gen(D,C)
##Generating the circle
x_circ= circ_gen(A,r)

#Plotting all lines
plt.plot(xBA[0,:],xBA[1,:],label='$Radius$')
plt.plot(xAD[0,:],xAD[1,:],label='$Radius$')
plt.plot(xBC[0,:],xBC[1,:],label='$Tangent$')
plt.plot(xDC[0,:],xDC[1,:],label='$Tangent$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((A,B,D,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','D','C']
for i, txt in enumerate(vert_labels):
        plt.annotate(txt, # this is the text
                                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                                                  textcoords="offset points", # how to position the text
                                                                   xytext=(0,10), # distance from text to points (x,y)
                                                                                    ha='center') # horizontal alignment can be left, right or center

        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend(loc='best')
        plt.grid()
        plt.axis('equal')

plt.savefig('/sdcard/gokul/matrices/circle/images/matrix.pdf')
subprocess.run(shlex.split("termux-open /sdcard/gokul/matrices/circle/images/matrix.pdf"))