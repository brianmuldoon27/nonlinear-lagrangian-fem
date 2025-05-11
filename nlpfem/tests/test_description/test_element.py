"""Tests for the bilinear quad element class"""

from lib2to3 import refactor
import numpy as np 
import casadi as csd 
import pytest
from nlpfem.description import BilinearQuadElement

# test the construction of a single element:
X = csd.SX.sym('X_0',4,2)
u = csd.SX.sym('u_0',4,2)
ref_pos = np.array([[0,0],
                    [1,0],
                    [1,1],
                    [0,1]])
u_ref = np.array([[0,0],
                  [1,0],
                  [1,0],
                  [0,0]])

dof_nums = np.array([0,1,2,3])
lame = np.array([100E6, 40E6])
node_nums = np.array([0,1,2,3])
density = 10000
element = BilinearQuadElement(num=0,
                              node_nums=node_nums,
                              node_pos=X,
                              dof_nums=dof_nums,
                              density=10000,
                              lame=lame)

# test properties 
assert np.isclose(node_nums, element.node_nums).all(), 'Element node numbers dont match.'


# test element mass matrix
Me = element.compute_mass_matrix()
F_Me = csd.Function('F_Me', [X], [Me])
Me_eval = F_Me(ref_pos)
assert np.sum(Me_eval.full().diagonal()) == 2 * density, "Mass matrix diagonal are wrong"


# test element stress divergence vector 
Re = element.compute_stress_divergence_vector(u_hat=u)
F_Re = csd.Function('F_Re', [X, u], [Re] )
Re_eval = F_Re(ref_pos, u_ref)
print(Re_eval)
# test element 



# check the 

print('done')
