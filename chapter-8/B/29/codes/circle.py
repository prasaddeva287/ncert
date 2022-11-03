import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as  LA

import sys  #for path to external scripts
sys.path.insert(0,'/sdcard/FWCmodule1/circle/code/CoordGeo') #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *
#if using termux
import subprocess
import shlex


r=5  #taken from given radius of circle
C=np.array(([2,-3])) 
S=np.array(([-3,2]))
V=np.array(([1,0],[0,1]))
u=np.array(([-2,3]))
f=-12

#directional vector
p=S-C
m=omat@p


#m,q,V,u,f

A,B=inter_pt(m,C,V,u,f) #q=C


##Generating the circle
c_circ= circ_gen(C,r)


##Generating all lines
x_CS = line_gen(C,S)
x_AB = line_gen(A,B)
x_SA = line_gen(S,A)


#Plotting all lines
plt.plot(x_CS[0,:],x_CS[1,:])
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_SA[0,:],x_SA[1,:],label='$Radius$')

#Plotting the circle
plt.plot(c_circ[0,:],c_circ[1,:],label='$Circle C$')


#generating new circle 'S'
R=LA.norm(A-S)
s_circ= circ_gen(S,R)
plt.plot(s_circ[0,:],s_circ[1,:],label='$Circle S$')
print("Raduis of S is :",round(R,2))


#Labeling the coordinates
tri_coords =np.vstack((C,S,A,B)).T

plt.scatter(tri_coords[0,:],tri_coords[1,:])
vert_labels = ['C','S','A','B']
for i,txt in enumerate(vert_labels):
	plt.annotate(txt,
			(tri_coords[0,i],tri_coords[1,i]),
			textcoords="offset points",
			xytext=(0,10),
			ha='center')

plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')

plt.savefig('/sdcard/FWCmodule1/circle/output.pdf')
subprocess.run(shlex.split("termux-open  /sdcard/FWCmodule1/circle/output.pdf"))

#plt.show()





