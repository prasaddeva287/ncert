import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys     #for path to external scripts
sys.path.insert(0,'/home/dell/matrix/CoordGeo/')

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Standard basis vectors
e1 = np.array((1,0)).reshape(2,1)
e2 = np.array((0,1)).reshape(2,1)

#Input parameters
r=4      #radius of the circle
h=np.array((4,5)).reshape(2,1)   #ext point
V = np.eye(2)
u = np.array((-2,-1)).reshape(2,1)
f =-11
O=-u.T  #center

#Intermediate parameters
f0 = np.abs(u.T@LA.inv(V)@u-f)
S = (V@h+u)@(V@h+u).T-(h.T@V@h+2*u.T@h+f)*V

#Eigenvalues and eigenvectors
D_vec,P = LA.eig(S)
lam1 = D_vec[0]
lam2 = D_vec[1]
p1 = P[:,1].reshape(2,1)
p2 = P[:,0].reshape(2,1)
D = np.diag(D_vec)
t1= np.sqrt(np.abs(D_vec))
negmat = np.block([e1,-e2])
t2 = negmat@t1

#Normal vectors to the tangents
n1 = P@t1
n2 = P@t2

den1 = n1.T@LA.inv(V)@n1
den2 = n2.T@LA.inv(V)@n2

k1 = np.sqrt(f0/(den1))
k2 = np.sqrt(f0/(den2))

#points of contact
#q11 = LA.inv(V)@((k1*n1-u.T).T)
q12 = LA.inv(V)@((-k1*n1-u.T).T)
q21 = LA.inv(V)@((k2*n2-u.T).T)
#q22 = LA.inv(V)@((-k2*n2-u.T).T)

#Generating all lines
xhq12 = line_gen(h,q12)
xhq21 = line_gen(h,q21)
xhq1 = line_gen(-u,q12)
xhq2 = line_gen(-u,q21)
xhq3 = line_gen(-u,h)

#Generating the circle
x_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(xhq12[0,:],xhq12[1,:],label='$Tangent1$')
plt.plot(xhq21[0,:],xhq21[1,:],label='$Tangent2$')
plt.plot(xhq1[0,:],xhq1[1,:],label='$Normal1$')
plt.plot(xhq2[0,:],xhq2[1,:],label='$Normal2$')
plt.plot(xhq3[0,:],xhq3[1,:],label='$Diagnol$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')

#Area of Quadrilateral
print("Area of Quadrilateral is:"+str(LA.norm(np.cross(q12.T-(-u.T),q12.T-h.T))))

#Labeling the coordinates
tri_coords = np.vstack((h.T,q12.T,q21.T,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['h','A','B','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), #this is point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') #horizontal alignment can be left,right,center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/home/dell/matrix/circ.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/Download/anusha1/python1/circle2.pdf"))
#else
plt.show()
