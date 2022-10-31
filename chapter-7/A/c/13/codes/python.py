#To find the incenter of a circle

#Python libraries for math and graphics
import numpy as np
import math 
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/hp/Tabassum/Python/CoordGeo')


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
A = np.array(([1,math.sqrt(3)]))
B = np.array(([0,0]))
C = np.array(([2,0]))

#computations
V1 = B-C
V2 = C-A
V3 = A-B
a = np.linalg.norm(V1)
b = np.linalg.norm(V2)
c = np.linalg.norm(V3)


#Solution by calculating angular bisectors equations
#D = a*(np.array(([np.cos(theta/2), np.sin(theta/2)])))
#E = b*(np.array(([np.cos(math.pi/2+theta/2), np.sin(math.pi/2+theta/2)])))
#F = c*(np.array(([np.cos(math.pi/2+math.pi+theta/2), np.sin(math.pi/2+math.pi+theta/2)])))

R1 = c/b #Ratio for D
R2 = c/a #Ratio for E
R3 = b/a #Ratio for F

D = (R1*C+B)/(R1+1)
E = (R2*B+A)/(R2+1)
F = (R3*C+A)/(R3+1)

a0 = D[1] - A[1]
b0 = A[0] - D[0]
c0 = a0*(A[0]) + b0*(A[1])

a1 = B[1] - F[1]
b1 = F[0] - B[0]
c1 = a1*(F[0]) + b1*(F[1])

G = np.array(([a0,b0],[a1,b1]))
H = np.array(([c0,c1]))

I = LA.solve(G,H)
print (I)



#solution2 by calculating centroid if equilateral triangle
I = (A+B+C)/3
print (I)


#Solution3 by directly using formula Incenter
I = (1/(a+b+c))*np.array(A*a + B*b +C*c)
print (I)

##Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_AC = line_gen(A,C)
#x_BI = line_gen(B,I)
#x_AI = line_gen(A,I)
#x_CI = line_gen(C,I)
x_BF = line_gen(B,F)
x_CE = line_gen(C,E)
x_AD = line_gen(A,D)

##Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_AC[0,:],x_AC[1,:])
#plt.plot(x_AI[0,:],x_AI[1,:])
#plt.plot(x_CI[0,:],x_CI[1,:])
#plt.plot(x_BI[0,:],x_BI[1,:])
plt.plot(x_BF[0,:],x_BF[1,:])
plt.plot(x_CE[0,:],x_CE[1,:])
plt.plot(x_AD[0,:],x_AD[1,:])


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,I,D,E,F)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','Incentre','D','E','F']
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

#if using termux
plt.savefig('/home/hp/Tabassum/Python/Incenter.pdf')
plt.show()
