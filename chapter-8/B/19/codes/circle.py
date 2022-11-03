#Python libraries for math and graphics

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random 

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/IIT_H/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

r=5
alpha=0

while(alpha<10):
    O=np.array(([alpha,2]))
    
    
    ##Generating the circle
    x_circ= circ_gen(O,r)
    
    
    #theta = 2*np.pi/3
    #np.random.seed(alpha)
    theta = 2*np.pi*np.random.sample()
   
    
    
    A = np.array([r*np.cos(theta), r*np.sin(theta)]) + O  #To take any points in a circle
    
    tri_coords = np.vstack((O,A)).T
    plt.scatter(tri_coords[0,:], tri_coords[1,:])
    vert_labels = ['O','A']
    for i, txt in enumerate(vert_labels):
        plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
   

    
    alpha += 2
  

    #Plotting the circle
    plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
    
    #generate the radius
    x_OA = line_gen(O,A)
    plt.plot(x_OA[0,:],x_OA[1,:],label='$Radius$')
   
    #Labeling the coordinates

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/IIT_H/sol/matrix2.pdf')
subprocess.run(shlex.split("termux-open '/sdcard//IIT_H/sol/matrix2.pdf'")) 
#else
#plt.show() 
