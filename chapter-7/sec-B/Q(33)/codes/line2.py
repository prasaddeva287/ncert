#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To verify areas of two triangles are equal


#Python libraries for math and graphics
import numpy as np
from pylab import *
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys                                          #for path to external scripts
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

#if using termux
import subprocess
import shlex
#end if

#Input parameters
O= np.array(([-1,-2]))
F= np.array(([[1,-1],[7,-1]]))
E= np.array(([-1,5]))
A= LA.inv(F)@E
print(A)

C= 2*O-A
print(C)
m=A-C
n=(m@np.array(([[0,1],[-1,0]])))
R=np.array(([[7,-1],m]))
H=np.array(([5,(m[0]*O[0]+m[1]*O[1])]))
D=LA.inv(R)@H
print(D)

B=A+C-D
print(B)


##Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_OD = line_gen(O,D)
x_CD = line_gen(C,D)
x_BD = line_gen(B,D)
x_AD = line_gen(A,D)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_BC[0,:],x_BC[1,:])#,label='$Diameter$')
plt.plot(x_CA[0,:],x_CA[1,:])#,label='$Diameter$')
plt.plot(x_OD[0,:],x_OD[1,:])#,label='$Diameter$')
plt.plot(x_CD[0,:],x_CD[1,:])#,label='$Diameter$')
plt.plot(x_BD[0,:],x_BD[1,:])#,label='$Diameter$')
plt.plot(x_AD[0,:],x_AD[1,:])#,label='$Diameter$')


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/dinesh/line2/line2.pdf')
subprocess.run(shlex.split("termux-open /sdcard/dinesh/line2/line2.pdf"))
#else
plt.show()
