import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          
sys.path.insert(0,'/sdcard/gokul/matrices/circle/codes/CoordGeo')   

from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

import subprocess
import shlex

lamda1=5
lamda2=9
V = np.array(([lamda1,0],[0,lamda2]))

u = np.array(([0,0]))
e1=np.array(([1,0]))
e2=np.array(([0,1]))
e3=np.array(([-1,0]))
e4=np.array(([0,-1]))

f =-45
f0=-f

e=np.sqrt(1-(lamda1/lamda2))

f1=e*(np.sqrt((f0)/(lamda2*(1-e**2))))
f2=-f1
F1=e*(np.sqrt((f0)/(lamda2*(1-e**2))))*e1
F2=e*(np.sqrt((f0)/(lamda2*(1-e**2))))*e3

l=2*np.sqrt((f0*lamda1))/lamda2
l1=l/2
l2=-l1

P=np.array(([f1,l1]))
Q=np.array(([f1,l2]))
S=np.array(([f2,l1]))
R=np.array(([f2,l2]))


n =  np.array(([10,15])) #normal vector
m =  omat@n #direction vector
c = 45
A= c/(n@e1)*e1 #x-intercept
D= c/(n@e2)*e2 #y-intercept
C= c/(n@e1)*e3 #x-intercept
B= c/(n@e2)*e4 #y-intercept
print(l)
v1=u-D
v2=u-A
ar_t1=0.5*np.linalg.norm((np.cross(v1,v2)))
ar=4*ar_t1
print("Area of quadrilateral is ", ar)

xAB= line_gen(A,B)
xBC = line_gen(B,C)
xCD = line_gen(C,D)
xDA = line_gen(D,A)
xAC = line_gen(A,C)
xDB = line_gen(D,B)

##Generating the circle
x_elli= ellipse_gen(np.sqrt(lamda2),np.sqrt(lamda1))

#Plotting all lines
plt.plot(xAB[0,:],xAB[1,:],label='$Tangent$')
plt.plot(xBC[0,:],xBC[1,:],label='$Tangent$')
plt.plot(xCD[0,:],xCD[1,:],label='$Tangent$')
plt.plot(xDA[0,:],xDA[1,:],label='$Tangent$')
plt.plot(xAC[0,:],xAC[1,:],label='$Tangent$')
plt.plot(xDB[0,:],xDB[1,:],label='$Tangent$')

#Plotting the circle
plt.plot(x_elli[0,:],x_elli[1,:],label='$Ellipse$')


#Labeling the coordinates
tri_coords = np.vstack((u,P,Q,A,B,C,D,P,Q,R,S,F1,F2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['u','P','Q','A','B','C','D','P','Q','R','S','F1','F2']
for i, txt in enumerate(vert_labels):
        plt.annotate(txt, # this is the text
                                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                                                  textcoords="offset points", # how to position the text
                                                                   xytext=(0,10), # distance from text to points (x,y)
                                                                                    ha='center') # horizontal alignment can be left, right or center

        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend(loc='best')
        plt.grid()
        plt.axis('equal')

plt.savefig('/sdcard/gokul/matrices/conics/images/matrix.pdf')
subprocess.run(shlex.split("termux-open /sdcard/gokul/matrices/conics/images/matrix.pdf"))