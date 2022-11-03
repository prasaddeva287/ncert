#Code by GVV Sharma (works on termux)
#February 12, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To find the centre of a circle given the end points of a diameter


#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/srinath/Documents/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if
C = np.array(([1,1],[1,-1]))
b = np.array(([0,0]))
A = np.array(([-8,8]))
B = np.array(([-2,2]))
e1 = np.array(([1,0]))
n1 = C[0,:]
n2 = C[1,:]
c1 = b[0]
c2 = b[1]
p = np.array(([-4,-4]))

#Solution vector
O = LA.solve(C,b)

#Direction vectors
m1 = omat@n1
m2 = omat@n2
#print("abc",m1,m2)

#Points on the lines
x1 = c1/(n1@e1)
A1 =  x1*e1
x2 = c2/(n2@e1)
A2 =  x2*e1
#print(x1,x2)

#Generating all lines
k1 = -10
k2 = 6
x_AB = line_dir_pt(m1,A1,k1,k2)
x_CD = line_dir_pt(m2,A2,k1,k2)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])#,#label='$Diameter$')
plt.plot(x_CD[0,:],x_CD[1,:])#,label='$Diameter$')

#Input parameters
#A = np.array(([-8,8]))
#B = np.array(([-2.5,2.5]))

#Centre and radius
c = np.array(([-9,1]))
r = 5*np.sqrt(2)

##Generating all lines
x_cA = line_gen(c,A)
x_cB = line_gen(c,B)

##Generating the circle
x_circ= circ_gen(c,r)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$chord$')
plt.plot(x_cA[0,:],x_cA[1,:],label='$line joining$')
plt.plot(x_cB[0,:],x_cB[1,:],label='$line joining$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((O,p,c,A,B)).T
#tri_coords1= c.T
#print(tri_coords,tri_coords1)
#print(tri_coords[0,0])
plt.scatter(tri_coords[0,:],tri_coords[1,:])
#plt.scatter(tri_coords1[0],tri_coords1[1])

vert_labels = ['O','p','c','A','B']
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
plt.savefig('/home/srinath/circle/matrix.pdf')
#subprocess.run(shlex.split("termux-open /home/srinath/Documents/matrix.pdf"))
#else
plt.show()






