#python codes for generation of circle
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
import numpy as np
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
#end if
# Given Diameter lines of the circles
# Line1: 2x-3y=5
# line2: 3x-4y=7

#Input parameters
n1 = np.array(([2,-3]))
n2=np.array(([3,-4]))#from given line equations
c = np.array(([5,7]))# from given line equations
e1 = np.array(([1,0]))#standard basis vector
e2 = np.array(([0,1]))#standard basis vector
A=np.block([[n1],[n2]])
#n1 = A[0,:]
#n2 = A[1,:]
#c1 = b[0]
#c2 = b[1]

Ar=np.pi*49#Given area of the circle

print("------------------------")
print("1. Center of the Circle")
#Solution vector
Center = LA.solve(A,c)
print(Center)

print("------------------------")
print("2. Radius of the Circle") 
import math
r = int(np.sqrt(Ar/(math.pi)))
print("radius", r)

print("------------------------")
print("3. f value of the circle")
f=((np.linalg.norm(Center))**2-r**2)
print("f =", f)

print("------------------------")
print("4. V matrix of the circle")
V= np.array(([e1],[e2]))
print("V =", V)

print("------------------------")
print("5. u matrix of the circle")
u=-Center
print("u=", u)

print("------------------------")
print("6. Equation of the Circle")
x=sym.Symbol('x')
y=sym.Symbol('y')
X=np.array((x,y))

#Circle equation X^T(V)+2(u^T)X+f=0
Cir_eq= (X.transpose()@X)+2*(u.transpose()@X)+f*(e1@ e1) #circle equation
print("{}=0".format(Cir_eq))
print("------------------------")


##Generating the circle
C = np.array(([1,-1]))
O = np.array(([0,0]))

x_circ= circ_gen(C,r)

#Plotting all lines
x = np.linspace(-13,13,14)
plt.plot(x, 0.667*x-1.667, '-g',label='$line: 2x-3y=5$')
plt.plot(x, 0.75*x-1.75,'-m',label='$line: 3x-4y=7$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:], '-r', label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((C,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['C(1,-1)', 'O']
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
#plt.savefig('/sdcard/Download/tabrez5/circlefig.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/Download/tabrez5/circlefig.pdf"))
#plt.savefig('/home/administrator/Assignment5/circlefig.pdf')
plt.show()

