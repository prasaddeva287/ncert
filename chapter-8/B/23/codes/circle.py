import math
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import subprocess
import shlex  
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
#Given Points
A = np.array(([1,0]))
B = np.array((2,3))
W = np.array((-1,0))
m = np.array((1,0))
#Finding Center
At = 2*A.T
Bt = 2*B.T
mt = m.T
i = -np.linalg.norm(A)**2
j = -np.linalg.norm(B)**2
k = -m@A
S = np.block([[At,1],[Bt,1],[mt,0]])
#S = np.array([[At[0],At[1],1],[Bt[0],Bt[1],1],[mt[0],mt[1],0]])


T = np.block([i,j,k])
P = LA.solve(S,T)
print("Solution vector p=",P)
u=np.array((P[0],P[1]))
print("u =",u)
#C = np.array(((-P[0],-round(P[1],2))))
C = np.block(-(u))
print("center = ",C)
f=P[2]
print("f = ",f)
r=math.sqrt(u@u-f)
print("Radius = ",round(r,2))
d=2*r
print("Diameter = ",round(d,2))

#Circle generation and plotting
x_circ = circ_gen(C,r)
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')	
x_AW=line_gen(A,W)                                                                    
plt.plot(x_AW[0,:],x_AW[1,:],label='$Tangent$')
tri_coords = np.vstack((A,B,C,W)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','W']
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
#plt.savefig('/sdcard/Download/codes/circle_assignment/cicle1.pdf')
#subprocess.run(shlex.split("termux-open  'storage/emulated/0/Download/codes/circle_assignment/circle1.pdf'"))
plt.show()
