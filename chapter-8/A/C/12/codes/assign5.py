

#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

#local imports
#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import circ_gen

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB
   
   #Generating points on a circle
def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ
   
#if using termux
import subprocess
import shlex
#end if

#Input parameters
#a = 2*mp.sin(theta)
alpha =60* np.pi/180
theta = alpha/2
a = 2*mp.sin(theta) # radius of the circle
d = a*mp.csc(theta) # generation of radius 
l = a*mp.cot(theta) #tangent generation
print(l)

e1 = np.array(([1,0]))

#i/p parameters for outer circle
r = 2


#Centre and point 
O = a*e1 #Centre
P = np.array(([0,0]))

#center and point{outer circle}
O = d*e1#center
S = np.array(([-2,0]))


#inner circle tangents 
Q = l*np.array(([mp.cos(theta),-mp.sin(theta)]))  #t1
R = l*np.array(([mp.cos(theta),mp.sin(theta)]))   #t2
R = np.array(R.tolist(), dtype=float)
Q = np.array(Q.tolist(), dtype=float)
O = np.array(O.tolist(), dtype=float)

#outer circle tangents
Q_1 = np.array(([1,-1.732]))  #t3
R_1 = np.array(([1,1.732]))   #t4
R_1 = np.array(R_1.tolist(), dtype=float)
Q_1 = np.array(Q_1.tolist(), dtype=float)
O = np.array(O.tolist(), dtype=float)


#tangents to inner circle
A = np.array([1.1,0])* np.array(([mp.cos(theta),-mp.sin(theta)]))
B = np.array([1.1,0])* np.array(([mp.cos(theta),-mp.sin(theta)]))
C = np.array([3,1.7])* np.array(([mp.cos(theta),mp.sin(theta)]))
D = np.array([3,-1.7])* np.array(([mp.cos(theta),mp.sin(theta)]))

##Generating all lines
xPQ = line_gen(P,Q)
xPR = line_gen(P,R)
xOP= line_gen(O,P)
xOR = line_gen(O,R)
xR_1A = line_gen(R_1,A)
xQ_1B = line_gen(Q_1,B)
xR_1C = line_gen(R_1,C)
xQ_1D = line_gen(Q_1,D)

##Generating all lines
xSQ_1 = line_gen(S,Q_1)
xSR_1 = line_gen(S,R_1)
xOS = line_gen(O,S)
xOR_1 = line_gen(O,R_1)

##Generating the circle
x_circ= circ_gen(O,a)
y_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(xPQ[0,:],xPQ[1,:],label='$T1$')
plt.plot(xPR[0,:],xPR[1,:],label='$T2$')
plt.plot(xOR[0,:],xOR[1,:],label='$Radius$')
plt.plot(xOP[0,:],xOP[1,:])#,label='$locus$')

#Plotting all lines
plt.plot(xSQ_1[0,:],xSQ_1[1,:],label='$T3$')
plt.plot(xSR_1[0,:],xSR_1[1,:],label='$T4$')
plt.plot(xOR_1[0,:],xOR_1[1,:],label='$Radius2$')
plt.plot(xOS[0,:],xOS[1,:],label='$locus2$')
plt.plot(xR_1A[0,:],xR_1A[1,:])#,label='$locus$')
plt.plot(xQ_1B [0,:],xQ_1B [1,:])#,label='$locus$')
plt.plot(xR_1C[0,:],xR_1C[1,:])#,label='$locus$')
plt.plot(xQ_1D[0,:],xQ_1D[1,:])#,label='$locus$')


#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:])#label='$Circle1$'
plt.plot(y_circ[0,:],y_circ[1,:])#label='$Circle2$'

#Labeling the coordinates
tri_coords = np.vstack((P,Q,R,O,S,Q_1,R_1,A,B,C,D)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q','R','O','S','Q1','R1','A','B','C','D']
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
#plt.savefig('/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-18.pdf')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-18.pdf"))
#else
plt.show()
