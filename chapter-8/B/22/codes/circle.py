

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/manoj/Documents/CoordGeo')         #path to my scripts
#Code by GVV Sharma (works on termux)
#February 12, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To find the centre of a circle given the end points of a diameter


#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
#A = np.array(([-6,3]))
#B = np.array(([6,4]))

#Centre and radius
#O = np.array(([0,0]))
#A = np.array(([4,0]))
#r = LA.norm(A-O)
#C = np.array(([4,0]))

a = 6
I = np.eye(2)               
u1 = np.array(([-a/2,0]))     
o1 = -u1                        
V1 = I                    
f1 = 0              
r1 = a/2                  
V2 = I
u2 = np.array(([0,0]))             
o2 = -u2                  
c = a                        
f2 = -c**2              
r2 = c                 

#for plotting


##Generating all lines
#x_OC = line_gen(O,C)

##Generating the circle
x_circ= circ_gen(o1,r1)
y_circ= circ_gen(o2,r2)


#Plotting all lines
#plt.plot(x_OC[0,:],x_OC[1,:],label='$C$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(y_circ[0,:],y_circ[1,:],label='$Circle$')

#Labeling the coordinates
tri_coords = np.vstack((o1,o2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['o1(a/2,0)','o2(0,0)']
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
plt.savefig('/home/manoj/git/FWC/Matrix/circle/figure.pdf')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-3.pdf"))
plt.show()







