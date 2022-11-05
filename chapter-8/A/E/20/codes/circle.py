#Code by GVV Sharma (works on termux)
#March 6, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To construct a circle and two tangents to it from a point outside


#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/module1/circle/CoordGeo')         #path to my scripts
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if
v = np.array(([2,-1.5],[-1.5,1]))
e, p = np.linalg.eig(v)
print("eigen value: ",e)
print("eigen vectors:",p)
e1 = np.sqrt(abs(e[0]))
e2 = np.sqrt(abs(e[1]))
e3 = np.sqrt(abs(e[0]))
e4 = -np.sqrt(abs(e[1]))
i = np.array(([e1,e2]))
j = np.array(([e3,e4]))
k1 = p@i
k2 = p@j
print("n1:",k1,"n2:",k2)
#Input parameters
r = 3
l1 = np.array(([k1[0],k1[1]]))
l2 = np.array(([k2[0],k2[1]]))
m1 = np.linalg.norm(l1)
m2 = np.linalg.norm(l2)
x = l1@l2
y = m1*m2
z = x/y
theta1 = np.arccos(z)
w = theta1 *180/np.pi
print(w)
#alpha = w *np.pi/180
theta = theta1/2

l = r*mp.cot(theta)
d = r*mp.csc(theta)
print(d)
e1 = np.array(([1,0]))
#Centre and point 
O = d*e1 #Centre
P = np.array(([0,0]))

Q = l*np.array(([mp.cos(theta),-mp.sin(theta)]))
R = l*np.array(([mp.cos(theta),mp.sin(theta)]))

R = np.array(R.tolist(), dtype=float)
Q = np.array(Q.tolist(), dtype=float)
O = np.array(O.tolist(), dtype=float)


##Generating all lines
xPQ = line_gen(P,Q)
xPR = line_gen(P,R)
xOP= line_gen(O,P)
xOQ = line_gen(O,Q)
xOR = line_gen(O,R)
##Generating the circle
x_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(xPQ[0,:],xPQ[1,:],label='$Tangent1$')
plt.plot(xPR[0,:],xPR[1,:],label='$Tangent2$')
plt.plot(xOQ[0,:],xOQ[1,:],label='$Radius$')
plt.plot(xOR[0,:],xOR[1,:],label='$Radius$')
plt.plot(xOP[0,:],xOP[1,:],'-.',label='$Radius$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((P,Q,R,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','B','A','C']
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
plt.savefig('/sdcard/Download/module1/circle/circle.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/module1/circle/circle.pdf"))
#else
#plt.show()
