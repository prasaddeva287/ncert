import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sympy as sym
import subprocess
import shlex
def ellipse_gen(a,b):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_ellipse = np.zeros((2,len))
	x_ellipse[0,:] = a*np.cos(theta)
	x_ellipse[1,:] = b*np.sin(theta)
	return x_ellipse
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
#Finding a and b 
U1=np.array([0,-2])
U2=np.array([-1,0])
u=np.array([0,0])
f1=0
f2=0
R1=math.sqrt(U1@U1-f1)#R=sqrt(u^T*U-f)
R2=math.sqrt(U2@U2-f2)
a=2*R1
b=2*R2
print("semi major axis=a=",a)
print("semi minor axis=b=",b)
#Finding V
f=sym.Symbol('f')
λ1=-f/a**2 #semi major axes =a=sqrt((u^T*V^-1*U-f/)λ1)
λ2=-f/b**2 #semi minor axes =b=sqrt((u^T*V^-1*U-f/)λ2)
V0=np.array(([λ1,0],[0,λ2]))
V=V0/-f
#To find equation of ellipse
x=sym.Symbol('x')
y=sym.Symbol('y')
X=np.array((x,y))
eq=(X.T@V@X)+2*(u.T@X)-1  #x^TVx + 2u^T x + f = 0
print("The equation of ellipse is {}=0".format(eq))
#For plotting
O=np.array([0,0])
P=np.array([4,0])
Q=np.array([0,2])
O1=np.array([0,1])
O2=np.array([2,0])
##Generating the diagram
x_ellipse=ellipse_gen(a,b)
x_PO = line_gen(P,O)
x_QO = line_gen(Q,O)
x_circ1=circ_gen(O2,R1)
x_circ2=circ_gen(O1,R2)

#Plotting the diagram
plt.plot(x_ellipse[0,:],x_ellipse[1,:],label='$ellipse$')
plt.plot(x_PO[0,:],x_PO[1,:],label='$line$')
plt.plot(x_QO[0,:],x_QO[1,:],label='$line$')
plt.plot(x_circ1[0,:],x_circ1[1,:],'g--',label='$Circle$')
plt.plot(x_circ2[0,:],x_circ2[1,:],'r--',label='$Circle$')

#Labeling the coordinates
tri_coords = np.vstack((O,P,Q)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','P','Q']
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
#using termux
plt.savefig('/sdcard/Download/codes/conics_assignment/co.pdf')
subprocess.run(shlex.split("termux-open  'storage/emulated/0/Download/codes/conics_assignment/co.pdf'"))
#plt.show()
