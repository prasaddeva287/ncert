import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys, os                                 
sys.path.insert(0,'/home/user/module1/matrix/line/CoordGeo')
#sys.path.insert(0,'/sdcard/Download/module1/circle/CoordGeo')  #path to my scripts
#local imports
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import parab_gen

#if using termux
import subprocess
import shlex
#end if

def affine_transform(P,c,x):
    return P@x + c
    
#Input parameters
V = np.array([[0,0],[0,1]])
u = np.array(([-2,0]))
o=np.array(([0, 0]))
f = 0
t1 = 2
t2 = -2
p = np.array(([t1**2, 2*t1]))
q = np.array(([t2**2, 2*t2]))
print(p,q,o)
a = o-p
b = q-o
print(a,b)
x = ((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)))
print(x)
y = np.arccos(x)
theta = y *180/np.pi
print(theta)
print("intersection of PQ and x-axis :",p[0])
#for plotting parabola
lamda,P = LA.eigh(V) 
if(lamda[1] == 0):      # If eigen value negative, present at start of lamda
    lamda = np.flip(lamda)       
    P = np.flip(P,axis=1)
eta = u@P[:,0] 
a = np.vstack((u.T + eta*P[:,0].T, V))     
b = np.hstack((-f, eta*P[:,0]-u))
center = LA.lstsq(a,b,rcond=None)[0]
O=center
n = np.sqrt(lamda[1])*P[:,0]
print(n)
#n = np.array(([0,1]))
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)  
print(c)    
F = (c*n - u)/lamda[1]
fl = LA.norm(F)   
x=12

num_points =50
delta = 2*np.abs(fl)/10
p_y = np.linspace(-10*np.abs(fl)-delta,10*np.abs(fl)+delta,num_points)
g = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y
##Generating all shapes
p_x = parab_gen(p_y,g)
p_std = np.vstack((p_x,p_y)).T
# generating all lines
x_op = line_gen(o,p)
x_oq = line_gen(o,q)
x_pq = line_gen(p,q)
plt.plot(x_op[0,:],x_op[1,:])
plt.plot(x_oq[0,:],x_oq[1,:])
plt.plot(x_pq[0,:],x_pq[1,:])
##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
#Plotting all shapes
plt.plot(p[0,:], p[1,:])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axhline(y=0,color='black')
plt.axvline(x=0,color='black')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('/sdcard/Download/module1/conic/main.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/Download/module1/conic/main.pdf"))
plt.show()
