import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
        #path to my scripts


#local imports
#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import *
#if using termux
import subprocess
import shlex
#end if
#from params import *

def dir_vec(A,B):
  return B-A

def norm_vec(A,B):
  return np.matmul(omat, dir_vec(A,B))

#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB


def ellipse_gen(a,b):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_ellipse = np.zeros((2,len))
	x_ellipse[0,:] = a*np.cos(theta)
	x_ellipse[1,:] = b*np.sin(theta)
	return x_ellipse
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100
y = np.linspace(-5,5,len)

#Given Points
lambda1=1/4
lambda2=1
f = -1
V = np.array(([lambda1,0],[0,lambda2]))
q = np.array(([4,0]))
a = 1/np.sqrt(lambda1)
b = 1/np.sqrt(lambda2)
l = np.array(([a,0]))
m = np.array(([0,b]))
d = np.array(([1,1]))
dl = (V@l).T
dm = (V@m).T
tan = np.block([[dl],[dm]])
p = LA.solve(tan,d)
P = np.diag(p)
Q = np.diag(q)
c = np.array(([0,0]))
A = np.array(([-a,b]))
B = np.array(([-a,-b]))
C = np.array((a,-b))
dA = np.block([[p@P],[q@Q]])
db = np.array(([1,1]))

#Ellipse parameters
d = LA.solve(dA,db)
x = np.sqrt(1/d[0])
y = np.sqrt(1/d[1])
print(x,y)
xStandardEllipse = ellipse_gen(a,b)
StandardEllipse = ellipse_gen(x,y)
#Major and Minor Axes
MajorStandard = np.array(([a,0]))
MinorStandard = np.array(([0,b]))
MajorStandard1 = np.array(([x,0]))
MinorStandard1= np.array(([0,y]))



#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

line1=line_gen(p,A)
line2=line_gen(A,B)
line3=line_gen(B,C)
line4=line_gen(C,p)


plt.plot(line1[0,:],line1[1,:],'r')
plt.plot(line2[0,:],line2[1,:],'b')
plt.plot(line3[0,:],line3[1,:],'g')
plt.plot(line4[0,:],line4[1,:],'r')


#Plotting the standard ellipse
plt.plot(xStandardEllipse[0,:],xStandardEllipse[1,:],label='Standard ellipse')
plt.plot(StandardEllipse[0,:],StandardEllipse[1,:])


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,p,q,c)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['$A$','$B$','$C$','$p$','$q$','$c$',]
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
plt.savefig('/sdcard/gowthami/figure/ellipse.pdf')
plt.savefig('/sdcard/gowthami/figure/ellipse.jpg')
subprocess.run(shlex.split("termux-open /sdcard/gowthami/figure/ellipse.png/ellipse.pdf"))
#else



