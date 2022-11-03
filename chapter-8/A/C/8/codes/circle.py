#Finding the centre of the circle
import numpy as np
import sympy as sp
from numpy import linalg as LA
from math import*
from numpy import poly1d
import matplotlib.pyplot as plt


import sys                                          #for path to external scripts
#sys.path.insert(0,'/sdcard/DCIM/Assignment-4/CoordGeo')   
sys.path.insert(0,'/sdcard/DCIM/Assignment-4/CoordGeo')       #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
#if using termux
import subprocess
import shlex
from math import *
#end if
x1,y1=2,-3
x2,y2=3,-4
c1,c2=5,7
n1=np.array([x1,y1])
n2=np.array([x2,y2])
n=np.array([n1,n2])
c=np.array([c1,c2])
center=LA.inv(n)@c
print("The centre of the circle is",center)
#from area of the circle, radius can be found
r=int(sqrt(154/pi))
print("Radius of the circle is ",r)

A = np.array(([2,-3],[3,-4]))#from given line equations
b = np.array(([5,7]))# from given line equations
e1 = np.array(([1,0]))#standard basis vector
e2 = np.array(([0,1]))#standard basis vector
x=sp.Symbol('x')
y=sp.Symbol('y')
X=np.array((x,y))



f=((LA.norm(center))**2-r**2)
print("f =", f)

print("------------------------")
print("4. V matrix of the circle")
V= np.block([[e1],[e2]])
print("V =", V)

print("------------------------")
print("5. u matrix of the circle")
u=-center
print("u=", u)

print("------------------------")
print("6. Equation of the Circle")
#Circle equation X^T(V)+2(u^T)X+f=0
Cir_eq= (X.T@X)+2*(u.T@X)+f #circle equation
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
plt.savefig('/sdcard/DCIM/Assignment-5/codes/circle.pdf')
plt.savefig('/sdcard/DCIM/Assignment-5/codes/circle.pdf')
subprocess.run(shlex.split("termux-open sdcard/DCIM/Assignment-5/codes/circle.pdf"))
#plt.show()

