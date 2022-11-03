import numpy as np
import math as m
import random as r
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sympy as sym
import math
import sympy
from sympy import Poly, roots, simplify,solve
from sympy import *
from sympy import roots, solve_poly_system


import sys                        #path to external scripts
sys.path.insert(0,'/home/user/Desktop/circles/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#input parameters
v1=np.array(([1,0],[0,0])).reshape(2,2)
u1=np.array([-4,0]).reshape(2,1)
f1=12
v2=np.array(([0,0],[0,1])).reshape(2,2)
u2=np.array([0,-7]).reshape(2,1)
f2=45
e1=np.array(([1,0])).reshape(2,1)
e2=np.array(([0,1])).reshape(2,1)
A=np.array(([1,0],[0,1]))
n1=e1
n2=e2
c1=np.array((2,5))
c2=np.array((6,5))
c3=np.array((2,9))
c4=np.array((6,9))
p1=LA.solve(A,c1)
p2=LA.solve(A,c2)
p3=LA.solve(A,c3)
p4=LA.solve(A,c4)
q1=(p1+p2)/2
q2=(p2+p4)/2
q3=(p3+p4)/2
q4=(p3+p1)/2
m1=omat@n1
m2=omat@n2
print(q1,q2,q3,q4)

#Plotting all lines
plt.axvline(x = 2)
plt.axvline(x = 6)
plt.axhline(y = 5)
plt.axhline(y = 9)

r=2
O=q1+q3
print(O.T/2)

O1=O.T/2



#Generating all lines



#Generating the circle
x_circ= circ_gen(O1,r)

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')

#Labeling the coordinates
tri_coords = np.vstack((q1,q2,q3,q4,O1)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','O']
for i, txt in enumerate(vert_labels):
   plt.annotate(txt,                                   # this is the text
                (tri_coords[0,i], tri_coords[1,i]),    # this is the point to label
                textcoords="offset points",            # how to position the text
                xytext=(5,-15),                      # distance from text to points (x,y)
                ha='left')                           # horizontal alignment can be left, right or center

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')

#if using termux
#plt.savefig('/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf')
#subprocess.run(shlex.split("termux-open '/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf'")) 
#else
plt.show()

