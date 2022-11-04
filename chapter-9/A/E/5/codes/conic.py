import numpy as np
from sympy import Symbol,solve,diff
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *
#if using termux
import subprocess           
import shlex
#end if



import sys                                          #for path to external scripts
sys.path.insert(0,'/home/shreyani/Documents/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs1 import conic_quad
#Generating points on a Given parabola
def parab_point(y,a):
	x = y**2/a
	return x
#Generating points on a  locus 
def Locus_point(y,a):
	x = (y**2)
	return x 
#Point of intersection of conic with line  
def inter_pt(m,q,V,u,f):
    a = m@V@m
    b = m@(V@q+u)
    c = conic_quad(q,V,u,f)
    l1,l2 =np.roots([a,2*b,c]) 
#    print(a,b,c)
    x1 = q+l1*m
    x2 = q+l2*m
    return x1


#equation of conic formula
#y**2 = 8*x ------given locus eqn

#Input parameters from given eqn

V = np.array(([0,0],[0,1]))
#print(V)
u = np.array([-2,0])
#print(u)
f = 0 


#given point
Q = np.array([0,0])


#symbols
x = Symbol('x')
y = Symbol('y')
X=np.array([x,y])


#section formula for 1:1 
#X = (Q+P)/2
P = ((4*X)-Q)/3
print(P)



q_locus = conic_quad(X,V,u,f)	# given locus
x_locus = conic_quad(P,V,u,f)   # obtained/req locus
print("------------------------------------------------")
print(q_locus,"= 0 --> given locus")
print(x_locus,"= 0 --> req locus")
#print("------------------------------------------------")
#print("---verification-----")
n = int(input("Input y-coordinate to generate point on abtained Locus: "))	
#Point on Given Locus
X = np.array([Locus_point(n,4),n])


#Point on given Parab 

m = dir_vec(Q,X)

P = np.array(inter_pt(m,X,V,u,f))

#print("verification of X ")
#comparison = X == (Q+(3*P))/4
	
#if(comparison.all()):
#	print("-----✓✓----locus is verified----✓✓----")
#else:
#	print("xxxxxxxxxYou have to work on itxxxxxxxxxx")
########################ploting############
#parameters
#Q = np.zeros(2)
#X = (Q+(3*P))/4

#generate the locus
y = np.linspace(-6, 6, 500)
x = (y**2)
#x = (9(y-8/9) ** 2)/4 + 2/9
k = np.linspace(-10, 10, 50000)
h =  (k**2)/4
#h = (k ** 2)/4
#generate line
x_QP = line_gen(Q,P)
#plotting the locus
plt.plot(h, k, label= "{}=0 given eq".format(q_locus))
plt.plot(x, y, label= "{}=0 msrd eq".format(x_locus/4))
#plot line
plt.plot(x_QP[0,:],x_QP[1,:])
#Labeling the coordinates
tri_coords = np.vstack((P,X,Q)).T
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['P','X','Q']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
                 (tri_coords[0, i], tri_coords[1, i]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Plot of locus equations')
    plt.legend(loc='best')
    plt.grid()
    plt.axis('equal')

##if using termux
#plt.savefig('/sdcard/Download/IITH-FWC-main/matrices/conic/conic.png')
#subprocess.run(shlex.split("termux-open /sdcard/Download/IITH-FWC-main/matrices/conic/conic.png"))          
#else     
plt.show()
