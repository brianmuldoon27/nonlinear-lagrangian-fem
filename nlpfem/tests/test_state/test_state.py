"""Method to test functionality of state objects"""
from nlpfem.state import PhysicsState
from nlpfem.description import TaylorImpactBar 

import numpy as np 

mesh = TaylorImpactBar(num_elem_x=20,
                       num_elem_y=4,
                       dx=1,
                       dy=0.2,
                       density=5000,
                       lame=np.array([1,1]),
                       wall_x=2)

# test init of state without uhat and lag_mult:
state1 = PhysicsState(mesh=mesh)

# test creation of state from values:
uhat_test_2 = np.ones((mesh.num_dofs,1))
lag_mult_test_2 = 2*np.ones((mesh.num_constraints,1))
state2 = PhysicsState(mesh=mesh,
                      uhat=uhat_test_2,
                      lag_mult=lag_mult_test_2)

# test the update of state1 to have state2 values:
state1.uhat = uhat_test_2
state1.lag_mult = lag_mult_test_2

# check values
assert np.isclose(np.linalg.norm(state1.uhat - state2.uhat),0), 'states are off'
assert np.isclose(np.linalg.norm(state1.lag_mult - state2.lag_mult),0), 'lagrange multipliers are off '


print('done')