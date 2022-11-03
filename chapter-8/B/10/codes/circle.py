#python codes for generation of circle
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sympy as sym
import shlex
import subprocess
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
# Given Diameter line and the circle
# Line1: y=x
# Circle1: x^2+y^2-2x=0

#Input parameters
A = np.array(([0,0]))#from given line and Circle equations
B = np.array(([1,1]))# from given line and Circle equations
e1 = np.array(([1,0]))#standard basis vector
e2 = np.array(([0,1]))#standard basis vector

print("------------------------")
print("1. Center of the Circle")
#Solution vector
C2 = (A+B)/2
print(C2)

print("------------------------")
print("2. Radius of the Circle") 
import math
r2 = (np.linalg.norm(B-C2))
print("r2=", r2)

print("------------------------")
print("3. f value of the circle")
f=((np.linalg.norm(-C2))**2-r2**2)
print("f =", f)

print("------------------------")
print("4. V matrix of the circle")
V= np.array(([e1],[e2]))
print("V =", V)

print("------------------------")
print("5. u matrix of the circle")
u=-C2
print("u=", u)

print("------------------------")
print("6. Equation of the Circle")
x=sym.Symbol('x')
y=sym.Symbol('y')
X=np.array((x,y))

#Circle equation X^T(V)+2(u^T)X+f=0
Cir_eq= (X.T@X)+2*(u.T@X)+f*(e1@ e1) #Required circle equation
print("{}=0".format(Cir_eq))
print("------------------------")


##Generating the circle
C1 = np.array(([1,0])) #center of given circle
C2 = np.array(([0.5,0.5]))#center of required circle
r1=1 #radius of given circle
r2=0.707 # radius of required circle


x_circ1= circ_gen(C1,r1)
x_circ2= circ_gen(C2,r2)

#Plotting all lines
x = np.linspace(-1,2,2)
plt.plot(x, x, '-g',label='$line: y=x$')

#Plotting the circle
plt.plot(x_circ1[0,:],x_circ1[1,:], '-r', label='$Circle:x^2+y^2-2x=0$')
plt.plot(x_circ2[0,:],x_circ2[1,:], '-m', label='$Circle:x^2+y^2-x-y=0$')

#Labeling the coordinates
tri_coords = np.vstack((C1,C2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['C1(1,0)','C2(0.5,0.5)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-16,0), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/sdcard/Github/Circle/Images/circle.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Github/Circle/Images/circle.pdf"))

