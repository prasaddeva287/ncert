import numpy as np
import math
from matplotlib import pyplot as plt, patches
from matplotlib.patches import Polygon
t1 = np.array([[-1,0], [0,0], [3,3*math.sqrt(3)]])
M = ((t1[1])-(t1[2]))
print('------------------------')
print("Direction vector of QR = ",M)
M=M/M[0]
m=M[1]
print('------------------------')
print("Slope of direction vector QR = ",'%.2f'%m)
theta=round(math.degrees(math.atan(m)))
print('------------------------')
print("Angle of QR with origin = ",theta)
pqr=int(180-theta)
print('------------------------')
print("Angle PQR = ",pqr)
r = math.radians(pqr)
q = math.tan(r)
qm = round(q,2)
print('------------------------')
print("Slope of the angular bisector of PQR = ", qm)
print('------------------------')
print("Equation of the angular bisecor is  Y = ",qm,"*X")
p1 = Polygon(t1, closed=True)
ax = plt.gca()
ax.add_patch(p1)
ax.set_xlim(1,10)
ax.set_ylim(1,10)
p1.set_facecolor('none')
p1.set_edgecolor('black')
plt.xlim(-2,4)
plt.ylim(0,6)
plt.ylabel("Y-axis ")
plt.xlabel("X-axis ")
plt.annotate("P", (-1,0))
plt.annotate("Q", (0,0))
plt.annotate("R", (3,5.19))
plt.annotate("M", (-0.7,1.21))
plt.plot([0,-1.5],[0,2.59], color='blue')
plt.scatter(-0.7,1.21, color='brown')
plt.plot([-0.7,-0.7],[0,1.21],color='orange')
plt.plot([-0.7,0.69],[1.21,1.21],color='orange')
plt.savefig('/sdcard/Github/Line/Images/diagram.png')
subprocess.run(shlex.split("termux-open /sdcard/Github/Line/Images/diagram.png"))