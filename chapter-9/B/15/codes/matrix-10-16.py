#Code by GVV Sharma (works on termux)
#March 4, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To find the intersection of a line with a circle


#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/ramesh/maths/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if


I =  np.eye(2)
e1 =  I[:,0]
#Input parameters
e=1/2
d=4
F=np.array(([0,0])).reshape(2,1)                          
e_1 =np.array(([1,0])).reshape(2,1) 
e_2=np.array(([0,1])).reshape(2,1) 
#ellipse  parameters
n= LA.norm(e_1)
o=np.transpose(e_1)
V =(n**2)*I-(e**2)*(e_1@o)
#print(e_1*o,e_1@o,o,(n**2))
u =(d*(e**2)*e_1)-(n**2)*F  
f = (n**2)*(LA.norm(F)**2)-(d**2)*(e**2)   
lambda_1=n**2
lambda_2=(1-e**2)*lambda_1
fo=((u.T)@(np.linalg.inv(V))@u)[0,0]-f
#print(n,V,u,f,fo,lambda_1,lambda_2,o)
a =np.sqrt(fo/lambda_2)
print("length of semi major axis :")
print(a)
b =np.sqrt(fo/lambda_1)
a1=e_1*a
b1=e_2*b
#genarating the lines
x_oa1=line_gen(F,a1)
##Generating the ellipse
x_ellipse= ellipse_gen(a,b)

#Plotting all lines
plt.plot(x_oa1[0,:],x_oa1[1,:],label='$Chord$')

#Plotting the elllipse
plt.plot(x_ellipse[0,:],x_ellipse[1,:],label='$Ellipse$')
a1 = a1.reshape(2)
b1 = b1.reshape(2)
#Labeling the coordinates
tri_coords = np.vstack((a1,b1)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['a1','b1']
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
plt.savefig('/sdcard/ramesh/maths/figs16.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/ramesh/maths/figs16.pdf"))
#else
#plt.show()






