#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/IITH/Assignment-1/MATRICES/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

I=np.eye(2)

#Circle parameters
V = I
u= np.array([1,2])
f = -3

#input parameters
A=-u                      #centre of the circle
P = np.array([1,0])       #given point
r = np.sqrt((u@u.T)-f)    #radius
Q = 2*A-P                 #required point
print("Q =",Q)
#r= LA.norm(P-A)
print(r)
#Generating the unit circle
xcirc1 = circ_gen(A,r)



##Plotting the circle
plt.plot(xcirc1[0,:],xcirc1[1,:],label='Circle')


#Generating all lines
xPA = line_gen(P,Q)
xPB = line_gen(P,A)

#Plotting all lines
#plt.plot(xPQ[0,:],xPQ[1,:],label='$Radius1$')

plt.plot(xPA[0,:],xPA[1,:],label='$Radius$')




#Labeling the coordinates
tri_coords = np.vstack((Q,A,P)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['Q','A','P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#
#if using termux
plt.savefig('/sdcard/IITH/Assignment-1/MATRICES/Circle/circleplot.pdf')
subprocess.run(shlex.split("termux-open /sdcard/IITH/Assignment-1/MATRICES/Circle/circleplot.pdf"))
#else
#plt.show()
