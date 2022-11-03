import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sympy import *

import sys
sys.path.insert(0,'/home/admin999/navya/matrix/CoordGeo')

from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

import subprocess
import shlex
a=6
b=7
p=3
q=4


A1=np.array(([1, 2*a, -b**2]))
A2=np.array(([1,2*p,-q**2]))
r1=np.roots(A1)
r2=np.roots(A2)

A=np.array([r1[0],r2[0]])
A3=np.array(A)

B=np.array([r1[1],r2[1]])

C=(A+B)/2
C1=np.array(C)
C=C.reshape(2,1)

A=A.reshape(2,1)
B=B.reshape(2,1)
radius=LA.norm(A-C)
f=(LA.norm(C))**2-radius
v=np.array([[1,0],[0,1]])
u=-v@C
x,y=symbols('x y')
eq=np.array([x,y])
eq=eq.reshape(2,1)
equation=eq.T@v@eq+2*u.T@eq+f
print(radius)
#print(C)
print(equation,"=0")
#print(C1)
x_CA=line_gen(C1,A3)
x_circ1=circ_gen(C1,radius)

plt.plot(x_circ1[0,:],x_circ1[1,:],label='Circle')
plt.plot(x_CA[0,:],x_CA[1,:])#,label='$Line')
tri_coords=np.vstack((C.T,A.T))
#print(tri_coords)
plt.scatter(tri_coords[:,0],tri_coords[:,1])
vert_labels = ['C(-6,-3)','A']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.savefig('/home/admin999/navya/matrix/circle.pdf')
plt.show()
