import numpy as np
import mpmath as mp
from matplotlib import pyplot as plt, patches
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sympy as sym

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

def dir_vec(A,B):
   return B-A
def norm_vec(A,B):
   return np.matmul(omat, dir_vec(A,B))
def circ_arc(O,r,theta1,theta2,len):
 theta = np.linspace(theta1,theta2,len)*np.pi/180
 x_circ = np.zeros((2,len))
 x_circ[0,:] = r*np.cos(theta)
 x_circ[1,:] = r*np.sin(theta)
 x_circ = (x_circ.T + O).T
 return x_circ
def circ_gen(O,r):
 len = 50
 theta = np.linspace(0,2*np.pi,len)
 x_circ = np.zeros((2,len))
 x_circ[0,:] = r*np.cos(theta)
 x_circ[1,:] = r*np.sin(theta)
 x_circ = (x_circ.T + O).T
 return x_circ
#if using termux
import subprocess
import shlex
#input parameters
cen=np.array((1,-2))
r=np.sqrt(2)
ao=np.sqrt(2)
oc=np.sqrt(2)
O=np.array((0,0))
ac=ao+oc
ac=2
print("Finding length of the square")
a=(ac/2)
print("a=",a)
print("-----------------------------------")
print("Finding the coordinates")
f=np.array([1,-1])
e=np.array([2,-2])
g=np.array([0,-2])
h=np.array([1,-3])
E=np.array([2,-2])
m1=np.array([0,1])
m2=np.array([1,0])



matM2 = np.block([[m1],[m2]]).T
lam = LA.solve(matM2,h-g)
A=h+lam[0]*m2
print("A=",A)

matM3 = np.block([[m1],[m2]]).T
lam = LA.solve(matM3,E-h)
B=h+lam[0]*m2
print("B=",B)

matM1 = np.block([[m1],[m2]]).T
lam = LA.solve(matM1,f-e)
C=f+lam[0]*m2
print("C=",C)

matM4 = np.block([[m1],[m2]]).T
lam = LA.solve(matM4,f-g)
D=g+lam[0]*m1
print("D=",D)



x_circ= circ_gen(cen,r)

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:], '-r', label='$Circle$')
#line gen
s=np.array([[2,-1],[0,-1],[0,-3],[2,-3]])
s1=Polygon(s, closed=True)
s1.set_edgecolor('red')
s1.set_facecolor('none')
ax=plt.gca()
ax.add_patch(s1)
ax.set_xlim(-1,3)
ax.set_ylim(-4,1)
plt.annotate("C(2,-1)",(2,-1))
plt.annotate("D(0,-1)",(0,-1))
plt.annotate("A(0,-3)",(0,-3))
plt.annotate("B(2,-3)",(2,-3))
plt.annotate("O(1,-2)",(1,-2))


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,cen)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['C(1,-2)', 'O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-16,0), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()

