"""Test file for ImplicitNewmark class"""

from unicodedata import ucd_3_2_0
import numpy as np 

from nlpfem.integration import ImplicitNewmark
from nlpfem.description import TaylorImpactBar
from nlpfem.state import PhysicsState

# create a test mesh
mesh = TaylorImpactBar(num_elem_x=20,
                       num_elem_y=4,
                       dx=1,
                       dy=0.2,
                       density=10000,
                       lame=np.array([100E6, 40E6]),
                       wall_x=2)

# create a test state object
state = PhysicsState(mesh=mesh)

# initialize a Newmark implicit solver
solver = ImplicitNewmark(mesh=mesh,
                         state=state,
                         beta=0.25,
                         gamma=0.5,
                         time_step=0.0002)

# check the time step and updating
assert solver.time_step == 0.0002, 'time step is off'
dt_new = 0.2
solver.time_step = dt_new 
assert solver.time_step == dt_new, 'updated time step is off'

# test updating of disp, vel, accel
u_new = np.ones_like(state.uhat)*2
v_new = np.ones_like(state.vhat)*3
a_new = np.ones_like(state.ahat)*4

# check updating of kineatmic vars
solver.assign_displacement(u_new=u_new)
solver.assign_velocity(v_new=v_new)
solver.assign_acceleration(a_new=a_new)
assert np.isclose(np.linalg.norm(solver.state.uhat - u_new),0)
assert np.isclose(np.linalg.norm(solver.state.vhat - v_new),0)
assert np.isclose(np.linalg.norm(solver.state.ahat - a_new),0)


# test residual map construction and eval
u0 = np.zeros_like(u_new)
v0  = np.zeros_like(v_new)
a0 = np.zeros_like(a_new)
res_test = solver.evaluate_residual(unp1=u0, 
                                    uhatn=u0,
                                    vhatn=v0,
                                    ahatn=a0)
assert np.isclose(np.linalg.norm(res_test), 0)
                            
# test the tangent stiffness map construction
K_test = solver.evaluate_tangent_stiffness(unp1=u0)
TOL = 10e-5
assert np.linalg.norm(K_test - K_test.T) < TOL, 'Tan stiffness is not symmetric'

# test the constraint vector building
c_vec = solver.constraint_vector

# test the extended system map building
ex_map = solver.extended_system_map
lm_test = np.zeros((mesh.num_constraints,1))
A,b = solver.evaluate_extended_system(u_k=u0, 
                                      uhatn=u0,
                                      vhatn=v0,
                                      ahatn=a0,
                                      lag_mult_k=lm_test)


print('hy')
