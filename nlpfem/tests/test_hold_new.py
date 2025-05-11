"""Test simulation to perform a hold test"""

import matplotlib.pyplot as plt
import numpy as np
from nlpfem.description import TaylorImpactBar
from nlpfem.state import PhysicsState
from nlpfem.integration import ImplicitNewmark
from nlpfem.visual.plt_utils import plot_mesh

# create my mesh
mesh = TaylorImpactBar(num_elem_x=20,
                       num_elem_y=4,
                       dx=1,
                       dy=0.2,
                       density=10E3,
                       lame=np.array([100E6, 40E6]),
                       wall_x=2)

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


# create a solver 
dt = 0.0002
solver = ImplicitNewmark(mesh=mesh,
                         state=init_state,
                         beta=0.25,
                         gamma=0.5,
                         time_step=dt,
                         nr_tol=10E-5)

# start time loop
fnames = []
# plot the mesh at time tnp1 mesh... 
plot_mesh(mesh=mesh, 
            time=0, 
            show_dof=False, 
            show_elem_num=False, 
            show_nn=False)

fn = f'gifs/new_hold/it_0.png'
fnames.append(fn)
plt.savefig(fn,dpi=300)


TOL = 10E-5
for n in range(100):

    # step in time
    time = (n+1)*solver.time_step

    print(f'=== Time Step = {n} === ')
    
    # initialize displacement solution for the time step
    unp1 = solver.state.uhat.copy()
    residual = solver.evaluate_residual(unp1=unp1,
                                        uhatn=solver.state.uhat,
                                        vhatn=solver.state.vhat,
                                        ahatn=solver.state.ahat)
    it = 0

    while (np.linalg.norm(residual) > TOL):
        print(f'Newton Raphson, iter={it}, residual={np.linalg.norm(residual)}')
        
        # form tangent stiffness
        K = solver.evaluate_tangent_stiffness(unp1=unp1)

        # get the residual
        residual = solver.evaluate_residual(unp1=unp1,
                                            uhatn=solver.state.uhat,
                                            vhatn=solver.state.vhat,
                                            ahatn=solver.state.ahat)
        
        # unpack the "active" dofs
        unp1_active = unp1[np.ix_(mesh._active_dofs, np.array([0]))].copy()
        K_active = K[np.ix_(mesh._active_dofs, mesh._active_dofs)]
        res_active = residual[np.ix_(mesh._active_dofs, np.array([0]))]
        
        # solve active system
        delta_u_active = -np.linalg.solve(K_active, res_active)
        unp1_active = unp1_active + delta_u_active

        # put the active part back into the total solution
        unp1 = unp1.copy() 
        unp1[np.ix_(mesh._active_dofs, np.array([0]))] = unp1_active
        
        it += 1
    
    unp1[np.ix_(mesh._active_dofs, np.array([0]))] = unp1_active 
    ahat_np1 = solver.beta_xl_update(unp1=unp1)
    solver.gamma_vel_update(ahat_np1=ahat_np1)

    # plot the mesh at time tnp1 mesh... 
    plot_mesh(mesh=mesh, 
              disp_vec=unp1,
              time=time,  
              show_dof=False, 
              show_elem_num=False, 
              show_nn=False,
              show_wire=True)
    
    # back out the Second Piola Kirchoff Matrix for each element
    # TODO: plot_mesh_stress()

    fn = f'gifs/hold/it_{i+1}.png'
    fnames.append(fn)
    plt.savefig(fn,dpi=300)

    # ITERATE IN TIME
    ahatn = ahat_np1 
    vhatn = vhat_np1 
    uhatn = unp1 