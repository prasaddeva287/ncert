#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/namrath/Downloads/fwc/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
R = np.array(([-6,0]))
B = np.array(([6,0]))

#Points on Circle
A = np.array(([-5.2,-3]))
#T = np.array(([4,4]))
#S = np.array(([6,6]))

#Centre and radius
O = (R+B)/2
r = LA.norm(R-O)

##Generating all lines
x_AB = line_gen(A,B)

##Generating the circle
o_circ = circ_gen(O,r)
i_circ = circ_gen(O,4)

#Computations
m = B - A
print(m)
#print(m)
l = (m.T)@A
#print(l)
k = (- l - (np.sqrt((l*l)-((LA.norm(m)**2)*((LA.norm(A)**2) - 4*4)))))/(LA.norm(m)**2)
print("k:",k)
C = A + k*m
print(C)

u = (- l + (np.sqrt((l*l)-((LA.norm(m)**2)*((LA.norm(A)**2) - 4*4)))))/(LA.norm(m)**2)
print("u:",u)
D = A + u*m
print(D)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])

#Plotting the circle
plt.plot(o_circ[0,:],o_circ[1,:],label='$OuterCircle$')

#Plotting the circle
plt.plot(i_circ[0,:],i_circ[1,:],label='$InnerCircle$')

#Labeling the coordinates
tri_coords = np.vstack((A,B,O,D,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','O','D','C']
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
plt.savefig('/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-16.pdf')
subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-16.pdf"))
#else
#plt.show()
#plt.show()
