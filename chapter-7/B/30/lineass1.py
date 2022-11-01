#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/matrices/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if



#Input parameters
P=  np.array(([2,2]))
Q=  np.array(([6,-1]))
R=  np.array(([7,3]))
S=  (Q+R)/2


#Directional vector
m=P-S
z=np.array(([0,1],[-1,0]))
n=np.matmul(z,m)


#point on lines
A=np.array(([1,-1]))

##Generating the line 
k1=-1.2
k2=0.5
x_AB = line_dir_pt(m,A,k1,k2)


xPQ = line_gen(P,Q)
xPR = line_gen(P,R)
xPS = line_gen(P,S)
xQR = line_gen(Q,R)

#print the equation
print(n[0],'*','x','+',n[1],'*','y','=',(n[0]*A[0]+n[1]*A[1]),sep=" ")


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='Line equation')

plt.plot(xPQ[0,:],xPQ[1,:],label='Side PQ')
plt.plot(xPR[0,:],xPR[1,:],label='Side PR')
plt.plot(xQR[0,:],xQR[1,:],label='Side QR')
plt.plot(xPS[0,:],xPS[1,:],label='Side PS')




#Labeling the coordinates
tri_coords = np.vstack((P,Q,R,S)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q','R','S']
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
plt.savefig('/sdcard/matrices/linefig.pdf')
subprocess.run(shlex.split("termux-open /sdcard/matrices/linefig.pdf"))
#else
#plt.show()

