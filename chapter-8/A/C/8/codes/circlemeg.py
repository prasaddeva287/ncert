#Code by GVV Sharma (works on termux)
#February 14, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To solve a system of linear equations 


#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *
import sys                                          #for path to external scripts
sys.path.insert(0,'/home/krishna/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
A = np.array(([2,3],[3,-1]))
b = np.array(([-1,4]))
e1 = np.array(([1,0]))
n1 = A[0,:]
n2 = A[1,:]
c1 = b[0]
c2 = b[1]

O =  np.array(([1,-1]))
#Solution vector
x = LA.solve(A,b)
r = 5
#Direction vectors
m1 = omat@n1
m2 = omat@n2

#Points on the lines
x1 = c1/(n1@e1)
A1 =  x1*e1
x2 = c2/(n2@e1)
A2 =  x2*e1

print(x)
x_circ= circ_gen(O,r)
#Generating all lines
k1=1.89
k2=-0.89
x_AB = line_dir_pt(m1,A1,k1,k2)
k1=1.9
k2=-1.25
x_CD = line_dir_pt(m2,A2,k1,k2)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='2x + 3y +1 = 0')
plt.plot(x_CD[0,:],x_CD[1,:],label='3x - y -4 = 0 ')

#Labeling the coordinates
#tri_coords = np.vstack((x)).T
tri_coords = x.T
#plt.scatter(tri_coords[0,:], tri_coords[1,:])
plt.scatter(tri_coords[0], tri_coords[1])
vert_labels = ['(1,-1)']
plt.plot(x_circ[0,:],x_circ[1,:],label='x^2+y^2−2x+2y−23=0')
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0], tri_coords[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-7.pdf')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-7.pdf"))
#else
plt.show()
