#Python libraries for math and graphics
import numpy as np
from pylab import *
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/module1/line/CoordGeo')
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
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#input parameters
O = np.array(([1,2]))
L1 = np.array(([1,1]))
L2 = np.array(([1,7]))
k1 = L1/np.linalg.norm(L1)
k2 = L2/np.linalg.norm(L2)
k = k1+k2
lamda = (-1/k[0]) 
A = O + lamda*k
C= 2*O-A
m1 = np.array([[1,7]])
m2 = np.array([[1,1]])
n1 = omat@m1.T
n2 = omat@m2.T
x = np.block([[[n1.T], [n2.T]]])
y = np.linalg.inv(x)
n1t = n1.T@A.T
n2t = n2.T@C.T
z = np.block([[[n1t], [n2t]]])
B=y@z
n3t = n1.T@C.T
n4t = n2.T@A.T
p = np.block([[[n3t.T],[n4t.T]]])
D = y@p
print(A)
print(B.T)
print(C)
print(D.T)
B = B.reshape((2,))
D = D.reshape((2,))
##Generating all lines
x_AB = line_gen(A,B)
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
plt.savefig('/sdcard/Download/module1/line/code1/main.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/module1/line/code1/main.pdf"))
#else
#plt.show()
