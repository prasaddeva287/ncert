import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from pylab import *
import sympy as sym
import math
import sympy 
from sympy import Poly, roots, simplify
from sympy import *
from sympy import roots, solve_poly_system
def parab_gen(y,a):
 x = y**2/a
 return x

def dir_vec(A,B):
  return B-A

def norm_vec(A,B):
  return np.matmul(omat,dir_vec(A,B))

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB

#Circle parameters

a=8
V=np.array(([0,0],[0,1]))
u=np.array([-4,0])
x=sym.Symbol('x')
y=sym.Symbol('y')
X=np.array([x,x+2])
f=0
n=np.array([1,-1])
F=X@V@np.transpose(X)+2*u@X+f
print("equation of parabola is : {}=0 ".format(F))
b= np.array([[0, 0], 
              [0, 1]])
v,w=eig(b)
p=np.array(([w[0][0],w[0][1]]))
print("eigen vector of 0 is : {}".format(p));
#finding the value of k
k=p@np.transpose(u)
print("The value of k is : {}".format(k))
r=u+k*n
g=np.array((r,[0,1]))
res=k*n-u
Y=np.array([-f,res[1]])
point1=np.linalg.solve(g,Y)
print("The point of contact of tangent to parabola is {}".format(np.linalg.solve(g,Y)))


n1=np.array([1,1])
r1=u+k*n1
g1=np.array((r1,[0,1]))
res=k*n1-u
Y1=np.array([-f,res[1]])
point2=np.linalg.solve(g1,Y1)
print("The point of contact of tangent to parabola is {}".format(np.linalg.solve(g1,Y1)))


dirvec=(point1-X)@np.transpose(X-point2)
print(dirvec)
from sympy import *
from sympy import roots, solve_poly_system
print(solve(dirvec, x))


def f(x):
    return np.array([x,x+2]);
print("The point outside the parabola from which the tangents are drawn perpendicular is : {}".format(f(-2)));






#equation of tangent to parabola at x1,y1
p_y = np.linspace(-6,6,100)
xparab = parab_gen(p_y,a)
x2=np.linspace(-6,2,100)
y2=x2+2;
plt.plot(x2,y2)
x3=np.linspace(-6,6,100)
y3=-x3-2;
plt.plot(x3,y3);

#Plotting the parabola
plt.plot(xparab,p_y,label='Parabola')
O=np.array([-2,0])
B=np.array([2,4])
A=np.array([2,-4])
plt.scatter(O[0],O[1])
plt.scatter(A[0],A[1])
plt.scatter(B[0],B[1])
vert_labels = ['O','A','B']
#for i, txt in enumerate(vert_labels):
plt.annotate("O", # this is the text
                 (O[0],O[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.annotate("A", # this is the text
                 (A[0],A[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.annotate("B", # this is the text
                 (B[0],B[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center





#for i, txt in enumerate(vert_labels):
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('/home/user/Desktop/conics/conic.pdf')
#subprocess.run(shlex.spli("termux-open /storage/emulated/0/github/cbse-papers/2020/math/12/solutions/figs/matrix-12-15.pdf"))
#else
plt.show()
