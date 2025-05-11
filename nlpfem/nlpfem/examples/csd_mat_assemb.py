"""Casadi matrix assembly test"""

import casadi as csd 
import numpy as np 

K11 = csd.SX.sym('K11',2,2)
K22 = csd.SX.sym('K22',2,2)

ind1 = np.array([0,1])
ind2 = np.array([2,3])

K_global = csd.SX.zeros((4,4))


K_global[np.ix_(ind1,ind1)] = K11
K_global[np.ix_(ind2,ind2)] = K22


Ke = csd.Function('Ke', [K11,K22], [K_global])

K11t = np.random.rand(2,2)
K22t = np.random.rand(2,2)

fe = Ke(K11t, K22t)

print(fe)

