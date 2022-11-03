


#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/FWC-main/FWC-main/matrix/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
r = 2
A = np.array(([0.5,2]))

n=np.array(([2,-1]))
#A1 = np.array(([1,1],[2.21,1]))
#b1 = np.array(([7,7.42]))
e1 = np.array(([1,0]))

k1 = -2
k2 = 2
AB = line_dir_pt(n,A,k1,k2)



#e1 = np.array(([1,0]))

#Centre and point 
u1 =np.array(([0,0])) #Centre
p=np.array(([0,2]))
plt.plot(AB[0,:],AB[1,:],label='$locus $')
##Generating the circle
x_circ= circ_gen(u1,r)
#Generating line
x_up= line_gen(u1,p)
#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle1$')
plt.plot(x_up[0,:],x_up[1,:],label='$radius=2$')

#Labeling the coordinates
tri_coords =u1.T
plt.scatter(tri_coords[0],tri_coords[1])
vert_labels = ['u1(0,0)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0],tri_coords[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-15), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')


#if using termux
plt.savefig('/sdcard/Download/FWC-main/FWC-main/matrix/circle1.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/FWC-main/FWC-main/matrix/circle1.pdf"))
#else
plt.show()



