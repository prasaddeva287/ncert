import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random 

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/ganga/matrix/CoordGeo')           #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if
I =  np.eye(2)
e1 =  I[:,0]
alpha=1
X=np.array(([-1,1]))
while(alpha<11):
    gamma=((-1+(2*alpha))**.5)-1
    B=np.array(([gamma,0])) 
    O=np.array(([gamma,alpha]))
    
    
    ##Generating the circle
    x_circ= circ_gen(O,alpha)
    theta = 1*np.pi/3
  
    R= np.array([alpha*np.cos(theta), alpha*np.sin(theta)]) + O #To take any points in a circle
    
    #R = np.array([h,(2*k)])  
    
    tri_coords = np.vstack((O,B,X,R)).T
    plt.scatter(tri_coords[0,:], tri_coords[1,:])
    vert_labels = ['O','B','X','R']
    for i, txt in enumerate(vert_labels):
        plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
   

    
    
  

    #Plotting the circle
    plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
    
    #generate the radius
    x_OR = line_gen(O,R)
    plt.plot(x_OR[0,:],x_OR[1,:],label='$Radius$')
    
    alpha += 2
    
m = e1
k1 = -10
k2 = 10
xline = line_dir_pt(m,B,k1,k2)
plt.plot(xline[0,:],xline[1,:],label='Tangent')
    #Labeling the coordinates

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/home/ganga/matrix/figs/circle.pdf')  
#subprocess.run(shlex.split("termux-open '/sdcard//Download/matrices/figs/circle1.pdf'")) 
#else
plt.show()
