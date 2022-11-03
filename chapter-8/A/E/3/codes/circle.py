
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/krishna/Krishna/python/codes/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
A = np.array(([1,1],[1,-1]))
P = np.array(([-4,3]))
b = np.array(([2,2]))
e1 = np.array(([1,0]))
n1 = A[0,:]
n2 = A[1,:]
c1 = b[0]
c2 = b[1]


#Solution vector
x = LA.solve(A,b)

#Direction vectors
m1 = omat@n1
m2 = omat@n2

#Points on the lines
x1 = c1/(n1@e1)
A1 =  x1*e1
x2 = c2/(n2@e1)
A2 =  x2*e1

print(x,x1,x2,P)

#Generating all lines
k1 = -15
k2 = 15
x_AB = line_dir_pt(m1,A1,k1,k2)
x_CD = line_dir_pt(m2,A2,k1,k2)

#Centre and radius
e1 = np.array(([-2.65,0]))
o=e1
e2 = np.array(([-17.35,0]))
o1=e2
r=3.28
r1=13.68
y=P-o
Y=np.linalg.norm(y)
print(Y)

##Generating the circle
x1_circ= circ_gen(o,r)
x2_circ= circ_gen(o1,r1)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$x+y=2$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$x-y=2$')
plt.plot(x1_circ[0,:],x1_circ[1,:],label='$Circle1$')
plt.plot(x2_circ[0,:],x2_circ[1,:],label='$Circle2$')

#Labeling the coordinates
#tri_coords = np.vstack((x)).T
tri_coords = np.stack((x,P,o,o1)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
#plt.scatter(tri_coords[0], tri_coords[1])
vert_labels = ['x','P','o','o1']

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
plt.savefig('/home/krishna/Krishna/python/figs/Circle.pdf')
#subprocess.run(shlex.split("termux-open "+os.path.join(script_dir, fig_relative)))
#else
plt.show()

