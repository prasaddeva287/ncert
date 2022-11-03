import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA
import sympy
import sys
import subprocess
import shlex
sys.path.insert(0,'/sdcard/Download/iith/python/Assignment-5/CoordGeo')

#a = np.array([-1,-k])
#b = np.array([0,-k])
#r1 = sqrt(k**2 - 5)
#r2 = sqrt(k**2 - k)
#pythagoreus theorem
#r1**2 + r2**2 = np.LA.norm(a-b)
#eq = 2*(k**2)-k-6
sympy.var('k')
p = sympy.solve((2*(k**2))- k - 6)
print("The possible values of k are",p)









