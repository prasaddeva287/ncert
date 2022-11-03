#To find the incenter of a circle

#Python libraries for math and graphics
import numpy as np
import math 
import matplotlib.pyplot as plt
from numpy import linalg as LA
import mpmath as mp

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/cir/CoordGeo')


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
#r1=int(input("Enter the radius : "))
r1=4
r2 = 3
O = np.array([1,3])
Q=np.array([4,-1])




##Generating all lines
x_circ1 = circ_gen(O,r1)
x_circ2 = circ_gen(Q,r2)
#xOB = line_gen(O,B)


##Plotting all lines
plt.plot(x_circ1[0,:],x_circ1[1,:],label='$circle1$')
plt.plot(x_circ2[0,:],x_circ2[1,:],label='$circle2$')
#plt.plot(xOB[0,:],xOB[1,:])




#Labeling the coordinates
tri_coords = np.vstack((O,Q)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','Q']
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

	#if using termux
plt.savefig('/sdcard/Download/cir/cirprb.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/cir/cirprb.pdf"))
	#plt.show()
