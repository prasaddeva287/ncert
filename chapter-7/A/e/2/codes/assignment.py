import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sympy import symbols, Eq, solve                                              #for equations
import sys                                          
sys.path.insert(0,'/home/sreshta/Rupa/matrix/Lines-Assignment/CoordGeo')
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

print("The area of triangle is 5.Two of its vertices are A(2,1) and B(3,-2). The third vertex C lies on y=x+3. Find C.")
x , y = symbols('x,y')                                                            #writing equations
eq1 = Eq((y-x),3)
print("Equation 1: ")
print(eq1)
eq2 = Eq((y+3*x),17)
print("Equation 2: ")
print(eq2)
print("The vertex of C is: ")
C=solve((eq1, eq2),(x, y))                                                        #values of x&y
print(C)

#Input parameters
A = np.array(([3,1],[1,-1]))
b = np.array(([17,-3]))
e1 = np.array(([1,0]))
n1 = A[0,:]
n2 = A[1,:]
c1 = b[0]
c2 = b[1]


#Solution vector
C = LA.solve(A,b)
print(C)
#given 
A=np.array(([2,1]))                                                               #given
B=np.array(([3,-2]))                                                              #given

#Generating line
x_AB = line_gen(A,B)
y_AC = line_gen(A,C)
z_BC = line_gen(B,C)

#Plotting line
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Line')
plt.plot(y_AC[0,:],y_AC[1,:])#,label='$Line')
plt.plot(z_BC[0,:],z_BC[1,:])#,label='$Line')
tri_coords = np.vstack((A,B,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

#For proving in python
v1=A-C
v2=A-B
V=ar_t1=0.5*np.linalg.norm((np.cross(v1,v2))) 
print("The area of triangle is:" , V)
if(V==5):
   print("Hence Verified")


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal') 

#for termux 
plt.savefig('/home/sreshta/Rupa/matrix/Lines-Assignment/line.pdf')
#for ubuntu
plt.show()
