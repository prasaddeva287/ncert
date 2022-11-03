import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/susi/Documents/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
#Input parameters

B=np.array([1,0])
I = np.eye(1)               
u2 = np.array([0,0])    
o2 = -u2                        
V2 = I                    
f2 = -9              
r2 = 3                  
V1 = I
#computation of center and radius of circle
r1=r2/2
x=-1/2
yp_1=np.sqrt(r2**2-1)
yp_2=yp_1/2
y=yp_2
u1=np.array([x,y])
e1 = np.array([1,0])
f=((np.linalg.norm(u2))**2-r2**2)
o1=-u1
r1=LA.norm(u1)


#Generating points on a circle
def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

#Generating the circle
x_circ = circ_gen(o2,r2)

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label="Circle1")

#Labelling the coordinates
tri_coords = np.vstack((o1,o2,B)).T
print(tri_coords)
print("center of the inner circle is:",(1.5,0.5))
print("the radius of the inner circle is:",1.58)
plt.scatter(tri_coords[0,:], tri_coords[1,:])
#plt.scatter(cp1)
vert_labels = ['o1','o2(0,0)','B(1,0)']
for i, txt in enumerate(vert_labels):
	plt.annotate(txt, #this is text
				 (tri_coords[0,i], tri_coords[1,i]), #this is the point to label
				textcoords="offset points" , # How to position the text
				xytext=(0,10),#Distance from the text to points (x,y)
				ha='center') # horizontal alignment can be left , right or center
#Generating points on a circle
def circ_gen_2(O,r):
	len = 50
	theta = 2*np.linspace(0,np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

#Generating the circle
x_circ_2 = circ_gen_2(o1,r1)
x_o2B = line_gen(o2,B)
plt.plot(x_o2B[0,:],x_o2B[1,:],label='$line$')

#Plotting the circle
plt.plot(x_circ_2[0,:],x_circ_2[1,:],label="Circle2")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.show()
#if using termux
#plt.savefig('/home/susi/Documents/figure1.pdf')
#subprocess.run(shlex.split("termux-open '/home/susi/Documents/figure1.pdf'")) 
#else

				
