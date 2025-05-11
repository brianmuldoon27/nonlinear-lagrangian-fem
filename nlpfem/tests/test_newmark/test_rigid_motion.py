"""Test simulation for rigid motion of a body approaching a wall
"""
import numpy as np
import casadi as csd 
import matplotlib.pyplot as plt

from nlpfem.description import TaylorImpactBar
from nlpfem.state import PhysicsState 
from nlpfem.integration import ImplicitNewmark
from nlpfem.visual import plot_mesh

# make a bar mesh
mesh = TaylorImpactBar(num_elem_x=5,
                       num_elem_y=1,
                       dx=1,
                       dy=0.2,
                       density=10E3,
                       lame=np.array([100E6, 40E6]),
                       wall_x=1+1)

# check the mass matrix
mass_matrix = csd.Function('M', [], [mesh.mass_matrix])
M = mass_matrix()['o0'].full()
a = M.sum()


# define initial velocity conditions
vhatn = []
for i in range(0, mesh.num_nodes):
    vi = np.array([[10], # x velocity
                   [0]]) # y velocity
    vhatn.append(vi)
vhatn = np.vstack(vhatn)

# construct the state object with initial condition prescribed
init_state = PhysicsState(mesh=mesh, 
                     vhat=vhatn)

# initiate an ImplicitNewmark solver
DT = 0.05
solver = ImplicitNewmark(mesh=mesh,
                         state=init_state,
                         beta=0.25,
                         gamma=0.5,
                         time_step=DT,
                         nr_tol=10E-5)

plot_mesh(mesh=mesh,
          show_dof=False, 
          show_nn=False,
          show_elem_num=False, 
          time=0)

N = 50 # time steps for the simulation
time = 0
times = [] 
times.append(time)

# assign initial value for disp
unp1 = init_state.uhat
for n in range(0, N):
    print(f'---- Time Step = {n} ----')

    # check the gaps before the solve
    gaps = solver.compute_gaps(uhat=unp1)
    gap_check = [np.isclose(g,0) for g in gaps.values()]

    # define initial guesses for the current time step
    unp1_guess = solver.state.uhat.copy()
    lag_mult_guess = np.ones_like(init_state.lag_mult)

    if any(gap_check): # the constraint is violated
        c = 0
        if c == 0:
            DT = 0.0007
            solver.assign_time_step(dt=DT, update_maps=True)
            c += 1

        # advance time
        time += DT
        times.append(time)
        print('Contact Detected -> Add constraints \n')

        # Impose v=0, a=0 on contact nodes...
        solver.state.vhat[np.ix_(mesh._contact_dofs, np.array([0]))] = np.zeros((2,1))
        solver.state.ahat[np.ix_(mesh._contact_dofs, np.array([0]))] = np.zeros((2,1))

        # solve the system with constraints imposed
        unp1, lag_mult = solver.compute_newton_raphson(unp1_guess=unp1_guess,
                                                       lag_mult_guess=lag_mult_guess,
                                                       in_contact=True)
        print('hey')

    else: # accept the time step result and advance time
        print('No Contact -> Accept Time Step \n')
        time += DT
        times.append(time)

        # ======= do a solve assuming no contact
        unp1, lag_mult = solver.compute_newton_raphson(unp1_guess=unp1_guess,
                                                        lag_mult_guess=lag_mult_guess,
                                                        in_contact=False)

    # do the newmark updates for velocity, and xl
    ahat_np1 = solver.beta_xl_update(unp1=unp1)
    vhat_np1 = solver.gamma_vel_update(ahat_np1=ahat_np1)

    # reassign values to state vector
    solver.assign_displacement(u_new=unp1)
    solver.assign_acceleration(a_new=ahat_np1)
    solver.assign_velocity(v_new=vhat_np1)
    solver.assign_lagrange_multipliers(lag_mult=lag_mult)

    # plot the updated mesh
    plot_mesh(mesh=mesh,
              disp_vec=solver.state.uhat,
              time=time,
              show_dof=False,
              show_nn=False,
              show_elem_num=False)

    plt.savefig(f'gifs/approach/time_step={n}')

    plt.close()

print('hey')


            
