import numpy as np
#import mpmath as mp
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
#from coeffs import *
import sys                                             #for path to external scripts
sys.path.insert(0,'/sdcard/Download/Line/CoordGeo')

from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if
#if using termux
import subprocess
import shlex
#end if

#Input parameters
A = np.array(([3,-4]))
B = np.array(([6,-8]))
b1 =-4
b2= 7
e1 = np.array(([1,0]))
#n1 = A[0,:]
#n2 = A[1,:]
#c1 = b[0]
#c2 = b[1]
P=np.array(([1,1.75]))
c3=np.linalg.norm(A)
D=(0.5*b2-b1)/c3
R=D/2
#direction vectors
m1=omat@A
m2=omat@B
#points on the line
x1=b1/(A@e1)
A1=x1*e1
x2=b2/(B@e1)
A2=x2*e1

C=np.array(([1.45,1.15]))
x_circ=circ_gen(C,R)
#generation of lines
k1=-4
k2=4
x_AB=line_dir_pt(m1,A1,k1,k2)
x_CD=line_dir_pt(m2,A2,k1,k2)
#lines plotting
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_CD[0,:],x_CD[1,:])
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle$')

#x_DC=line_gen(D,C)
#plt.plot(x_DC[0,:],x_DC[1,:],color='r')

##Labeling the coordinates
tri_coords = np.vstack((C,P)).T
#print('coor:',tri_coords)
plt.scatter(tri_coords[0,:],tri_coords[1,:])
vert_labels = ['C','P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
#
plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.xlim(-1,3)
plt.ylim(-1,3)
plt.savefig('/sdcard/Download/Circle/figure/fig6.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/Circle/figure/fig6.pdf"))
plt.show()
