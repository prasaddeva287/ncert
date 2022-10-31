
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/mat lab/linie/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
A = np.array(([1,2]))
c = np.array(([10,-5]))
b = np.array([2,1])
e1 = np.array(([1,0]))
i = np.array([4,-5])
n1 = A[0,:]
n2 = A[-1,:]
n3 = A[-2,:]


c1 = b[0]
c2 = b[1]

k=np.block([n1,n2,n3])
print(k)
#Solution vector
x = LA.solve(A,b)


m1 = omat@n1
m2 = omat@n2
m3 = omat@n3
#Points on the lines   
x1 = c1/(n1@e1)   
A1 =  x1*e1                            
x2 = c2/(n2@e1)                         
A2 =  x2*e1


print(x)

#Generating all lines
k1 = -15
k2 =5
x_AB = line_dir_pt(m1,A1,k1,k2)
x_CD = line_dir_pt(m2,A2,k1,k2)
x_EF = line_dir_pt(m3,x,k1,k2)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_CD[0,:],x_CD[1,:])#,label='$Diameter$')

#Labeling the coordinates
#tri_coords = np.vstack((x)).T
tri_coords = x.T
#plt.scatter(tri_coords[0,:], tri_coords[1,:])
plt.scatter(tri_coords[0], tri_coords[1])
vert_labels = ['x']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0], tri_coords[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/mat lab/linie/CoordGeo/figs2.pdf')
subprocess.run(shlex.split("termux-open /sdcard/mat lab/linie/CoordGeo/figs2.pdf"))
#else
#plt.show()
  
