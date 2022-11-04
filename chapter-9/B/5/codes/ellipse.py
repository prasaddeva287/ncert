#python codes for generation of Ellipse and its equation
import numpy as np
import mpmath as mp
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
#Generating points on an ellipse
def ellipse_gen(a,b):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_ellipse = np.zeros((2,len))
	x_ellipse[0,:] = a*np.cos(theta)
	x_ellipse[1,:] = b*np.sin(theta)
	return x_ellipse

#if using termux
import subprocess
import shlex
import math
#end if


#Input parameters
e=0.5 #eccentricty
u = np.array(([0,0]))# Origin as center of ellipse
c=4 #from the given directrix x=4
e1 = np.array(([1,0]))#standard basis vector
e2 = np.array(([0,1]))#standard basis vector
O=np.array((0,0))
#n1 = A[0,:]
#n2 = A[1,:]
#c1 = b[0]
#c2 = b[1]

print("------------------------")
print("1. vector n of the Ellipse")
n = np.array((1,0)) #found fro the directrix x=4 or (1,0)x=4
print("vector n =", n)

print("------------------------")
print("2. vector V of the Ellipse") 
I=np.array(([e1,e2]))
V=(((np.linalg.norm(n))**2)*I)-((e**2)*(n*n.transpose()))*I
print(V)

print("------------------------")
print("3. Vector u of the Ellipse")
u=-O@V
print(u)

print("------------------------")
print("4. focus point of the Ellipse")
lambda2=1 #from Vector V
F= ((c*(e**2)*n-u)/lambda2)
print("F =", F)

print("------------------------")
print("5. f value of the Ellipse")
f=((np.linalg.norm(n))**2)*((np.linalg.norm(F))**2)-(c**2)*(e**2)
print("f=", f)

print("------------------------")
print("6. Equation of the Circle")
x=sym.Symbol('x')
y=sym.Symbol('y')
X=np.array((x,y))

#Ellipse equation X^T(V)+2(u^T)X+f=0
Elps_eq= (X.transpose()@V@X)+2*(u.transpose()@X)+f*(e1@ e1) #Ellipse equation
print("{}=0".format(Elps_eq))
print("------------------------")



##plotting##
O = np.array([0,0])
F1 =  F
F2 = -F
Maj=2
Min=np.sqrt(3)

#Plotting all lines
x = np.linspace(-4,3,4)
plt.axvline(x=4,  color="g", label="Directrix x=4")
plt.axvline(x=-4, color="m", label="Directrix x=-4")

ellipse = ellipse_gen(Maj,Min)
plt.plot(ellipse[0,:],ellipse[1,:], '-r', label='$Ellipse:0.75x^2+y^2-3=0$')

tri_coords = np.vstack((O,F1,F2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','F1(1, 0)','F2(-1, 0)']
for i, txt in enumerate(vert_labels):
        plt.annotate(txt, # this is the text
                (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
        plt.xlabel('$x-axis$')
        plt.ylabel('$y-axis$')
        plt.legend(loc='best')
        plt.grid()
        plt.axis('equal')
#plt.savefig('/sdcard/Download/tabrez6/ellipsefig.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/Download/tabrez6/ellipsefig.pdf"))
plt.savefig('/home/administrator/Assignment6/ellipsefig.pdf')
plt.show()

