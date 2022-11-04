import numpy as np
import matplotlib.pyplot as plt

import math 
from numpy import linalg as LA

import sys, os
script_dir = os.path.dirname(__file__)
lib_relative = '../../coord/circle'
sys.path.insert(0,os.path.join(script_dir, lib_relative))

#local imports
from funcir import *

simlen = 200
#Standard parabola
y = np.linspace(-6,6,simlen)
x = parab_gen(y)

#Parabola points
#Standard Parabola Vertex
O = np.array([0,0])

#Focus
F= np.array([2,0])
V = np.array([[0,0],[0,1]])
u = np.array([-2,0])
#Point on parabola
a1 = 2
a2 = 2*(2**0.5)

A = np.array([a1,a2])

n = A@V + u
#print(n)
mu = (-2*LA.norm(n))/(n.T@V@n)

B = A + mu*n
print(B)



#Plotting the parabola
plt.plot(x,y,label='Standard Parabola')

#Plotting the directrix
#plt.plot(x,D[1]*np.ones(simlen),label='Directrix')


x_AB = line_gen(A,B)

x_AO = line_gen(A,O)
x_OB = line_gen(O,B)

plt.plot(x_AB[0,:],x_AB[1,:])

plt.plot(x_AO[0,:],x_AO[1,:])
plt.plot(x_OB[0,:],x_OB[1,:])
#Labeling the coordinates
parab_coords = np.vstack((O,F, A, B)).T
plt.scatter(parab_coords[0,:], parab_coords[1,:])
vert_labels = ['O','F','A', 'B']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (parab_coords[0,i], parab_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.axis('equal')

#plt.savefig('plot_con.png')

#OA & OB are perpendicular
#print(A@B)
#slope of AB
m = B-A
slope = m[1]/m[0]
print('slope of AB = ',slope)



