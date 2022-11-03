#Python libraries for math and graphics

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random 
import subprocess
import shlex

import sys                                         
sys.path.insert(0,"/home/chandana/Downloads/fwc-main/CoordGeo")         

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

I=np.eye(2)
e1=I[:,0]

#circle parameters
theta =90*np.pi/180
theta1=0*np.pi/180
theta2=30*np.pi/180
O=np.array(([1,0]))
r=1
A=np.array(([r*np.cos(theta1+theta),r*np.sin(theta1)]))
B=np.array(([r*np.cos(theta1),r*np.sin(-theta)]))
C=np.array(([r*np.cos(theta1+2*theta),r*np.sin(theta1+2*theta)]))
M1=(A+B)/2
M2=(B+C)/2
r2=np.linalg.norm(M1)
#locus equation
print(r2,end=" ")

#circle
x_circ=circ_gen(O,r)
x_circ1=circ_gen(O,r2)

#circleplot
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circles$')
tri_coords = np.vstack((A,B,O,M1,M2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','O','M1','M2']
plt.plot(x_circ1[0,:],x_circ1[1,:],'r--',label='$Circles$')
#Generating the line
m=-e1
k1=-10
k2=10
xAB=line_gen(A,B)
xOA=line_gen(O,A)
xOB=line_gen(O,B)
xOM1=line_gen(O,M1)
xOM2=line_gen(O,M2)
xBC=line_gen(B,C)
xOC=line_gen(O,C)



#Generating the circle 
x_circ1=circ_gen(O,r)
x_circ2=circ_gen(O,r2)
#plotting all the lines
plt.plot(xAB[0,:],xAB[1,:],label='Chord')
plt.plot(xBC[0,:],xBC[1,:],label='Chord')
plt.plot(xOB[0,:],xOB[1,:],label='radius')
plt.plot(xOC[0,:],xOC[1,:],label='radius')
plt.plot(xOA[0,:],xOA[1,:],label='radius')
plt.plot(xOM1[0,:],xOM1[1,:],label='chord bisector')
plt.plot(xOM2[0,:],xOM2[1,:],label='chord bisector')



#locus of midpoint of chord of the circle which subtends a right angle at the origin
#here locus is nothing but equation of the the points (x,y) and (0,0) which forms  a perpenducluar 
#from the figure

#labeling the coordinates
tri_coords = np.vstack((O,A,B,C,M1,M2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O(0,0)','A','B','C','M1','M2']
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
plt.show() 
plt.savefig('/sdcard/Download/c2_fwc/trunk/circle_assignment/docs/circle.png')
subprocess.run(shlex.split("termux-open '/sdcard/Download/c2_fwc/trunk/circle_assignment/docs/main.pdf' "))
