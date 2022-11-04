import numpy as np
import matplotlib.pyplot as plt
from math import *
from numpy import linalg as LA
import sympy
import sys
import subprocess
import shlex
sys.path.insert(0,'/sdcard/Download/iith/python/Assignment-5/CoordGeo')
#system imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

# solution #
from sympy import Eq, Symbol, solve

#consider f0/lambda = p
#using length of minor axis = 2*sqrt(p)
p = 16
q = 1/p
#using distance between foci = 6
e = Symbol('e')
exp = 9*(1 - (e**2))*(q**2) - (e**2)*q
E = solve(exp)
for j in E:
    if j>0:
        print("The eccentricity is %.1f" % j)
        #b = 4
        #a = int(sqrt(b**2/(1 - (j**2))))
        l = j
        break
b = 4
a = int(sqrt(b**2/(1 - (l**2))))

##plotting##
O = np.array([0,0])
A1 = np.array([a,0])
B1 = np.array([0,b])
A2 = np.array([-a,0])
B2 = np.array([0,-b])
F1 = np.array([a*l,0])
F2 = np.array([-a*l,0])

ma = line_gen(A1,A2)
mb = line_gen(B1,B2)
ff = line_gen(F1,F2)
ellipse = ellipse_gen(a,b)
plt.plot(ma[0,:],ma[1,:],label='$Major axis$')
plt.plot(mb[0,:],mb[1,:],label='$Minor axis$')
plt.plot(ff[0,:],ff[1,:],label='$dist bw foci$')
plt.plot(ellipse[0,:],ellipse[1,:],label='$Ellipse$')

tri_coords = np.vstack((O,A1,B1,A2,B2,F1,F2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','A1','B1','A2','B2','F1','F2']
for i, txt in enumerate(vert_labels):
        plt.annotate(txt, # this is the text
                (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend(loc='best')
        plt.grid()
        plt.axis('equal')

plt.savefig('/sdcard/Download/iith/python/Assignment-6/figure6.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/iith/python/Assignment-6/figure6.pdf"))
