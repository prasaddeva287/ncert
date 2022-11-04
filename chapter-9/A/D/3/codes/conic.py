## Conic 

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import mpmath as mp

import sys                                          #for path to external scripts
sys.path.insert(0, '/home/bhavani/Documents/CoordGeo')

#if using termux
import subprocess
import shlex
#end if

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB


def ellipse_gen(a,b):
	len = 100
	theta = np.linspace(0,2*np.pi,len)
	x_ellipse = np.zeros((2,len))
	x_ellipse[0,:] = a*np.cos(theta)
	x_ellipse[1,:] = b*np.sin(theta)
	return x_ellipse
	

a = 1/4
b = 1/9
u = np.array([0,0])
V = np.array(([b,0],[0,a]))
V1 = np.linalg.inv(V)
print("inverse of V",V1)
f = -(a*b)

#A1 = np.array([a,0])
#A2 =-A1
#B1 = np.array([0,b])
#B2 = -B1

x = np.linspace(-1,1,100)
y = (8/9)*x	
plt.plot(x,y)

n = np.array([8,-9])
k1 = np.sqrt(((u@V1@u) - f)/(n@V1@n))
k2 = -np.sqrt(((u@V1@u) - f)/(n@V1@n))
print(k1,k2) 
q1 = V1@(k1*n-u)	
q2 = V1@(k2*n-u)	
print("q1",q1)
print("q2",q2)

y1 = (8*x-5)/9
plt.plot(x,y1)

y2 = (5+8*x)/9
plt.plot(x,y2)

 

#Generating all lines
#x_XY = line_gen(X,Y)

#Generating the ellipse
x_ellipse = ellipse_gen(np.sqrt(a),np.sqrt(b))

#plotting all lines
#plt.plot(x_XY[0,:],x_XY[1,:],label = '$Diameter$')


#plotting the circle
#plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(x_ellipse[0,:],x_ellipse[1,:])


#Labeling the coordinates
tri_coords = np.vstack((q1,q2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['q1','q2']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid()
plt.axis('equal')

#if using termux
plt.savefig('/home/bhavani/Documents/matrix/matrix_conic/conic1.pdf')
#subprocess.run(shlex.split("termux-open /home/bhavani/Documents/circle1.pdf"))
#else
plt.show()
