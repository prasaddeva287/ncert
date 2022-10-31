#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/krishna/Krishna/python/codes/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
P=np.array(([2,3])) #Point
n=np.array(([1,1])) #normal vector
c=7
d=4 #distance
e1=np.array(([1,0]))
x1 = c/(n@e1) #X-intercept
A1 =  x1*e1

#Direction vector
m1=omat@n 


u=(np.linalg.norm(m1))**2
v=((m1)@(P-A1))
w=(v**2)
y=(np.linalg.norm(m1))**2
z=((np.linalg.norm(P-A1)**2)-d**2)
z1=(w)-(y*z)
z2=np.sqrt(z1)
lamda=(v+z2)/u
Q1= A1+((lamda)*m1)
Q2= A1-((lamda)*m1)


def norm_vec(P,Q1):
  return np.matmul(omat, dir_vec(P,Q1))
def norm_vec(P,Q2):
  return np.matmul(omat, dir_vec(P,Q2))

#Generating all lines
k1 = -15
k2 = 8
AB = line_dir_pt(m1,A1,k1,k2)
CD = line_dir_pt(dir_vec(P,Q1),Q1,k1,k2)
EF = line_dir_pt(dir_vec(P,Q2),Q2,k1,k2)


#intersection of two lines
Q3= line_intersect(n,A1,norm_vec(P,Q1),Q1)
Q4 = line_intersect(n,A1,norm_vec(P,Q2),Q2)

#Slope for lines
e3=np.array([[1],[0]])
e4=np.array([[0],[1]])
m1=P-Q2
k1=m1@e4/m1@e3
print(k1)
m2=P-Q3
k2=m2@e4/m2@e3
print(k2)

#Plotting all lines
plt.plot(AB[0,:],AB[1,:])#,label='$line1$'
plt.plot(CD[0,:],CD[1,:])#,label='$line2$'
plt.plot(EF[0,:],EF[1,:])#,label='$line3$'

#Labeling the coordinates
tri_coords = np.vstack((P,Q3,Q4)).T
#tri_coords = P.T
#plt.scatter(tri_coords[0], tri_coords[1])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q3','Q4']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')




