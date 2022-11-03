import numpy as np
import math 
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/sinkona/Documents/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Centre and radius
u = np.array(([2,4]))
r = np.array([5]) 

#Generating the line
x = np.linspace(-3,7,100)
for i in range(-35,15):
	y = (3*x-i)/4
	plt.plot(x, y, '-r')
	
#x = np.linspace(-2,8,100)
#y = (3*x)/4
#plt.plot(x, y, '-r',label='3x-4y=0')

#Generating the circle
x_circ= circ_gen(u,r)

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')

#Labeling the coordinates
tri_coords = np.vstack((u))
plt.scatter(tri_coords[0,:],tri_coords[1,:])
vert_labels = ['u[2,4]']
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
plt.show()
