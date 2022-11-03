#Code by GVV Sharma (works on termux)
#March 1, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To construct a circle and two tangents to it from a point outside
#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/ramesh/maths/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import*
from conics.funcs import *
#if using termux
import subprocess
import shlex
#end if
#Input parameters
I =  np.eye(2)
e1 =  I[:,0]
#input parameters
B= np.array(([2,1]))
#circle parameters
V = I
u = np.array(([-1,-3]))
f = 6
A =np.array(([1,3]))#center
r=2 #radius of first circle
#line parameters
n=B-A
m =omat@n #direction vector
print(m)
O =np.transpose(n)
print(O)
c =O@A
q = c/(n@e1)*e1 #x-intercept
#r=sqrt((LA.norm(c)+A)
#Points of intersection
#of line with circle
C,D = inter_pt(m,q,V,u,f)
print(C,D)
##Generating all lines
xAB = line_gen(A,B)
xAC = line_gen(A,C)
xCB = line_gen(C,B)
xAD = line_gen(A,D)
#radius of the circle2
r2 =LA.norm(C-B)
print("the radius of second circle :")
print(r2)

##Generating the circle
x_circ= circ_gen(A,r)
x1_circ= circ_gen(B,r2)


#Plotting all lines
plt.plot(xAB[0,:],xAB[1,:],label='$Tangent1$')
plt.plot(xAC[0,:],xAC[1,:],label='$Tangent2$')
plt.plot(xCB[0,:],xCB[1,:],label='$Tangent1$')
plt.plot(xAD[0,:],xAD[1,:],label='$Tangent1$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(x1_circ[0,:],x1_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D']
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
plt.savefig('/sdcard/ramesh/maths/figs13.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/ramesh/maths/figs13.pdf"))
#else
#plt.show()




