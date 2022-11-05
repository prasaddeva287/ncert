import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys          #for path to external scripts
#path to my scripts
sys.path.insert(0,'/home/hp/Tabassum/Python/CoordGeo')


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if


##Standard basis vectors
e1 = np.array((1,0)).reshape(2,1)
e2 = np.array((0,1)).reshape(2,1)
#Input parameters
r  = 2
r2 = 2*r
h = np.array((0,r2)).reshape(2,1)
h1 = np.array((0,r2))
V = np.eye(2)
u = np.zeros((2,1))
f =-r**2
S = (V@h+u)@(V@h+u).T-(h.T@V@h+2*u.T@h+f)*V
O = np.array(([0,0]))
##Intermediate parameters
#
f0 = np.abs(f+u.T@LA.inv(V)@u)
#
##Eigenvalues and eigenvectors
D_vec,P = LA.eig(S)
lam1 = D_vec[0]
lam2 = D_vec[1]
p1 = P[:,1].reshape(2,1)
p2 = P[:,0].reshape(2,1)
D = np.diag(D_vec)
#
t1= np.sqrt(np.abs(D_vec))
negmat = np.block([e1,-e2])
t2 = negmat@t1
#
##Normal vectors to the conic
n1 = P@t1
n2 = P@t2
#
den1 = n1.T@LA.inv(V)@n1
den2 = n2.T@LA.inv(V)@n2
#
k1 = np.sqrt(f0/(den1))
k2 = np.sqrt(f0/(den2))
#
q11 = LA.inv(V)@((k1*n1-u.T).T)
q12 = LA.inv(V)@((-k1*n1-u.T).T)
q21 = LA.inv(V)@((k2*n2-u.T).T)
q22 = LA.inv(V)@((-k2*n2-u.T).T)
#
####
####To prove that the Centroid lies on the inner circle
a = LA.norm(q11.T)
b = LA.norm(q22.T)
c = LA.norm(h1)
#print(a,c)

f1 = -a #On substituting q11 on inner circle 
f2 = -b #On substituting q22 on inner circle

#centroid of triangle 
C = (h1+q11.T+q22.T)/3

k = LA.norm(C)
print (k,f1)

x = k+f1 # if centroid lies on inner circle,it should satisfy ||C||+f1 = 0
print(x)

##Generating all lines
xhq11 = line_gen(h,q11)
xhq22 = line_gen(h,q22)
xhq33 = line_gen(q11,q22)
#
##Generating the circle
x_circ= circ_gen(O,r)
x_circ2= circ_gen(O,r2)

##Plotting all lines
plt.plot(xhq11[0,:],xhq11[1,:],label='$Tangent1$')
plt.plot(xhq22[0,:],xhq22[1,:],label='$Tangent2$')
plt.plot(xhq33[0,:],xhq33[1,:],label='')



#
#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$1')
plt.plot(x_circ2[0,:],x_circ2[1,:],label='$Circle2$')

#
#Labeling the coordinates
tri_coords = np.vstack((q11.T,q12.T,q21.T,q22.T,h.T,C,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['q11','q12','q21','q22','h','C','O']
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
#
#if using termux
#plt.savefig('//matrix-10-13.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-13.pdf"))
plt.show()
