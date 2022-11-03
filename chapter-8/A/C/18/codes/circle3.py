#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
#sys.path.insert(0,'/storage/emulated/0/github/cbse-papers/CoordGeo')         #path to my scripts
sys.path.insert(0,'/sdcard/circle/CoordGeo')


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if


#Standard basis vectors
e1 = np.array((1,0)).reshape(2,1)
e2 = np.array((0,1)).reshape(2,1)


I =  np.eye(2)
e3 =  I[:,1]
#e1 =  I[:,0]
#Input parameters

n =  np.array(([5,-2])) #normal vector
m =  omat@n #direction vector
c = -6
q = (c/(m@e1))*e3 #x-intercept
q=q.reshape(2,1)




#Input parameters
r  = 4#radius of the circle
#d =10
#theta=np.pi/3
#h = np.array((-d,0)).reshape(2,1)
#h11=np.sin(theta/2) #*np.array(([mp.cos(theta),mp.sin(theta)]))
#h=-r/h11*e1
V = np.eye(2)
u = np.array(([3,3])).reshape(2,1)

f =-2
S = (V@q+u)@(V@q+u).T-(q.T@V@q+2*u.T@q+f)*V

##Centre and point 
#u = np.array(([0,0]))
O = np.array((-3,-3)).reshape(2,1)
#p = np.array(([-3.66,0.923]))
#Intermediate parameters

f0 = np.abs(f+u.T@LA.inv(V)@u)

#Eigenvalues and eigenvectors
D_vec,P = LA.eig(S)
#print(D_vec,P)
lam1 = D_vec[0]
lam2 = D_vec[1]
p1 = P[:,1].reshape(2,1)
p2 = P[:,0].reshape(2,1)
D = np.diag(D_vec)
t1= np.sqrt(np.abs(D_vec))
negmat = np.block([e1,-e2])
t2 = negmat@t1

#Normal vectors to the conic
n1 = P@t1
n2 = P@t2
#print("t1=",t1,"t2=",t2,"p=",P)
#kappa
den1 = n1.T@LA.inv(V)@n1
den2 = n2.T@LA.inv(V)@n2

k1 = np.sqrt(f0/(den1))
k2 = np.sqrt(f0/(den2))


q21 = LA.inv(V)@((k2*n2-u.T).T)



D=LA.norm(q-P)
print(D)
#print(q11,q12,q21,q22)

#
##Generating all lines
xhq12 = line_gen(q,q21)
#xhq22 = line_gen(h,q22)
#
##Generating the circlei
x_circ= circ_gen(O.T,r)
#
##Plotting all lines
plt.plot(xhq12[0,:],xhq12[1,:],label='$Tangent1$')
#plt.plot(xhq22[0,:],xhq22[1,:],label='$Tangent2$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
#
#
#Labeling the coordinates
tri_coords = np.vstack((q21.T,q.T,O.T))
#print(tri_coords)
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['P','q','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
            (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#
#if using termux
plt.savefig('/sdcard/circle/fig.pdf')
subprocess.run(shlex.split("termux-open /sdcard/circle/fig.pdf"))
#else
plt.show()
#
#
#
#
#
#
#
