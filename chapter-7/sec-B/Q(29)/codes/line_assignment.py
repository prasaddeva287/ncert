#Code by GVV Sharma (works on termux)
#February 16, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To verify basic proportionality theorem


#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sys                                          #for path to external scripts
sys.path.insert(0,'/home/chinni/prathyusha/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
P=np.array([1,0])
Q=np.array([0,1])
R=np.array([1,1])
B=np.array([0,0])

C = 2*P-B
A= 2*Q-B

i=A-B
j=B-C
k=A-C

v1=(np.linalg.norm(i))
v2=(np.linalg.norm(j))
v3=(np.linalg.norm(k))

X=((v1*C) + (v2*A) + (v3*B))/(v1+v2+v3)
print("The x-coordinate of the incentre is", X[0])

l=A-X
m=X-B
n=X-C
z1=l@i
z2=l@k
v11=(np.linalg.norm(l))*(np.linalg.norm(i))
v22=(np.linalg.norm(l))*(np.linalg.norm(k))

angleBAX=np.arccos((z1)/(v11))
angleXAC=np.arccos((z2)/(v22))
print("angleBAX=", math.degrees(angleBAX))
print("angleXAC=", math.degrees(angleXAC))
print("Linesegment AX bisects the angle BAC")

z11=i@k
v12=(np.linalg.norm(i))*(np.linalg.norm(k))

angleBAC=np.arccos((z11)/(v12))
print("BAC=",math.degrees(angleBAC))


##Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(C,B)
x_CA = line_gen(A,C)
x_AX = line_gen(A,X)
x_BX = line_gen(B,X)
x_CX = line_gen(C,X)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_BC[0,:],x_BC[1,:])#,label='$Diameter$')
plt.plot(x_CA[0,:],x_CA[1,:])#,label='$Diameter$')
plt.plot(x_AX[0,:],x_AX[1,:])#,label='$Diameter$')
plt.plot(x_BX[0,:],x_BX[1,:])#,label='$Diameter$')
plt.plot(x_CX[0,:],x_CX[1,:])#,label='$Diameter$')



#Labeling the coordinates
tri_coords = np.vstack((A,B,C,P,Q,R,X)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','P','Q','R','X']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='right') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/home/chinni/prathyusha/matrix-10-10.pdf')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-10.pdf"))
#else
plt.show()





