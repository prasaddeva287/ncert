import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/sdcard/fwc/matrices/CoordGeo')

#import subprocess
#import shlex

r=2
e1=np.array(([1,0]))
e2=np.array(([0,1]))
d=2
m=np.array(([1,1]))
C=d*m
O=np.array(([0,0]))
A=np.array(([0,6.8]))
B=np.array(([6.9,0]))
P=(A+B)/2

def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB

Y=line_dir_pt(e1,O,-2,8)
X=line_dir_pt(e2,O,-2,8)
x_circ= circ_gen(C,r)
x_AB=line_gen(A,B)
x_CP=line_gen(C,P)
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(Y[0,:],Y[1,:],'g',label='$y=0$')
plt.plot(X[0,:],X[1,:],'g',label='$x=0$')
plt.plot(x_AB[0,:],x_AB[1,:],'g')
plt.plot(x_CP[0,:],x_CP[1,:])

#Labeling the coordinates
tri_coords = np.vstack((O,C,A,B,P)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','C','A','B','P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-5,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
plt.savefig('/sdcard/fwc/matrices/CoordGeo/circle_assignment/cir.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/fwc/matrices/circles/cir.pdf"))
