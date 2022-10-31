
# A,B,C are vertices of a triangle 

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

#input parameters
A=np.array([1,1])
E=np.array([-1,2])
F=np.array([3,2])


B=2*E-A
print("The vertex B is : ",B)

C=2*F-A
print("The vertex C is : ",C)

D=(B+C)/2

#Centroid=(A+B+C)/3
G=(A+B+C)/3
print("The centroid of a given triangle is : ",np.round(G,2))


#Gnenerate line points
x_AB = line_gen(A,B)
x_BC = line_gen(C,B)
x_CA = line_gen(A,C)
x_EC = line_gen(E,C)
x_FB = line_gen(F,B)
x_AD = line_gen(D,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_CA[0,:],x_CA[1,:])
plt.plot(x_EC[0,:],x_EC[1,:])
plt.plot(x_FB[0,:],x_FB[1,:])
plt.plot(x_AD[0,:],x_AD[1,:])

#Labelling the coordinates
tri_coords = np.vstack((A,B,C,G,D,E,F)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels= ['A(1,1)','B(-3,3)','C(5,3)','G(1,2.33)','D','E','F']
for i, txt in enumerate(vert_labels):
	plt.annotate(txt, #this is text
				 (tri_coords[0,i], tri_coords[1,i]), #this is the point to label
				textcoords="offset points" , # How to position the text
				xytext=(0,10),#Distance from the text to points (x,y)
				ha='center') # horizontal alignment can be left , right or center
plt.xlabel("X")
plt.ylabel("Y")
#plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.show()
