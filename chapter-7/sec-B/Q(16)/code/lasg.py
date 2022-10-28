#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          
#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A+lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB


#if using termux
#import subprocess
#import shlex
#end if
print("The line equation of a given intercept point is 4X +3Y=24")

##Generating all lines
A=np.array(([3,4])).reshape(2,1)#given point 
e1=np.array(([1,0])).reshape(2,1)#basis vector 
e2=np.array(([0,1])).reshape(2,1)#basis vector 
B=np.array(([0,2*e2.T@A])).reshape(2,1)
C=np.array([2*e1.T@A,0]).reshape(2,1)
x_BC=line_gen(B,C)#using linegen we draw a line 
#Plotting all lines
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
#plt.plot(x_CB[0,:],x_CB[1,:],label='$BC$')
#plt.plot(x1,y1)
x1=[3,6,0]
y1=[4,0,8]
pts1=['A','C','B']
plt.scatter(x1,y1)
#Labeling the coordinates
#tri_coords = np.vstack((A)).T
#plt.scatter(tri_coords[0,:], tri_coords[0,:])
vert_labels = ['A(3,4)','B(0,8)','C(6,0)']
for i, txt in enumerate(vert_labels):
   plt.annotate(txt, # this is the text
               (x1[i], y1[i]), # this is the point to label
               textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
               ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$X-axis$')
plt.ylabel('$Y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.axhline(y=0,color='black')
plt.axvline(x=0,color='black')

#if using termux
#plt.savefig("/sdcard/Download/codes/line/line.pdf")
#subprocess.run(shlex.split("termux-open '/sdcard/Download/codes/line/line.pdf'")) 
#else
plt.show()
