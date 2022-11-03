import numpy as np
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
#To generate condition
a,b,x = sym.symbols('a b x')
e_2 = np.array(([0,1]))
A = np.array(([a,b/2]))
C = np.array(([a/2,b/4]))
f = 0
U = -C
V = np.array(([0,0],[0,1]))
y = -e_2@A
P = np.array(([x,y]))
eq = P@P + 2*(U@P)+f
d = sym.discriminant(eq)
print("The condition is")
print(sym.StrictGreaterThan(d,0))

#To generate figure
p = 1
q = 1/np.sqrt(2)
C1 = np.array(([p/2,q/4]))
A1 = np.array(([p,q/2]))
r = np.linalg.norm(C1 - A1)
coeff = [1,-p,q**2/2]
X1 = np.roots(coeff)
P1 = np.array(([X1[0],-q/2]))
P2 = np.array(([X1[1],-q/2]))
M1 = (A1 + P1)/2
M2 = (A1+P2)/2
AP1 = line_gen(A1,P1)
AP2 = line_gen(A1,P2)
x_circ= circ_gen(C1,r)

#Plotting all lines
plt.plot(AP1[0,:],AP1[1,:],label='$AP1$')
plt.plot(AP2[0,:],AP2[1,:],label='$AP2$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')



#Labeling the coordinates
tri_coords = np.vstack((P1,P2,M1,M2,A1,C1)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P1','P2','M1','M2','A','C']
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
plt.savefig('/sdcard/iithfwc/trunk/circle/para.pdf')
plt.show()
