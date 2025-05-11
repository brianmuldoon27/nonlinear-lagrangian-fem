"""Script to solve the impact problem with release"""

from venv import create
import numpy as np 
import matplotlib.pyplot as plt 
import casadi as csd
import scipy 
from scipy import sparse
from scipy.sparse import linalg
from nlpfem.visual import plot_mesh, make_gif
from nlpfem.description import TaylorImpactBar

# problem set up
# x length
LX = 1 

# y length
LY = 0.2

# number of x elems
NENX = 20

# number of y elems
NENY = 4

DENSITY = 10000
LAME = np.array([100E6, 40E6])
h = LY/NENY
K = LAME[0] + 2/3*LAME[1]
E = 9*K * (K - LAME[0]) / (3*K - LAME[0])
c = np.sqrt(E/DENSITY)
dt_cr = h/c 

# make a mesh
mesh = TaylorImpactBar(num_elem_x=NENX,
                       num_elem_y=NENY,
                       dx=LX,
                       dy=LY,
                       density=DENSITY,
                       lame=LAME,
                       wall_x=2)

# ======= AUTO DIFF MASS MATRIX
M = mesh.mass_matrix
F_M = csd.Function('F_M', [], [M])
M_eval = F_M()['o0']

# ======= TEST ASSEMBLY OF GLOBAL STRESS DIVERGENCE VECTOR with casadi 
# create a vertically stacked vector of displacement dof symbols for each node 
un_states = []
for n in range(0, mesh.num_nodes):
    un = csd.SX.sym(f'u_{n}',2,1)
    un_states.append(un)
uhatnp1_sym = csd.vertcat(*un_states)
uhat_ref = np.zeros((2 * mesh.num_nodes, 1))

# ======= AUTODIFF RESIDUAL
R = mesh.compute_stress_divergence_vector(uhat=uhatnp1_sym)

# evaluate the stress divergence in the reference configuraiton (should be zero... )
F_R = csd.Function('F_R', [uhatnp1_sym], [R])
R_eval = F_R(uhat_ref).full()

# define the residual function 
def create_residual_map(dt, beta, uhatn, vhatn, ahatn, Fnp1):
    """Returns a residual function in terms of the displacement params we need to solve for a tnp1.
    
    Inputs:
        dt: (numeric scalar) the time step of the simulation
        beta: (numeric scalar) the newmark beta param
        uhatn: (2*num_nodes x 1) the displacement dof values at time tn 
        vhatn: (2*num_nodes x 1) The velocity dof values at time tn 
        ahatn: (2*num_nodes x 1) The acceleration dof values at time tn 
    Returns:
        residual: (2*num_nodes x 1) The residual as a function of autodiff parameters unp1 
    """
    residual = 1/(beta*dt**2) * M @ uhatnp1_sym + R - M @ ((uhatn + vhatn * dt) * (1/(beta*dt**2)) + (1-2*beta)/(2*beta) * ahatn) - Fnp1
    F_res = csd.Function('F_res', [uhatnp1_sym, uhatn, vhatn, ahatn, Fnp1], [residual])
    return F_res 

def create_constraint_map(mesh):
    """Method to build the constraint vector for the system.
    """
    c_entries = []
    for c_node in mesh._contact_nodes:
        # get reference position for the contact node
        X_ref = mesh.get_ref_node_position(target=c_node)

        # get dofs associated with the contact node
        cn_dofs = mesh.node_dof_map[c_node]

        # get the symbol representing the contact displacement dof (x direction)
        uc_sym = uhatnp1_sym[cn_dofs[0]]

        # get the u_bar
        u_bar = mesh.wall_x - X_ref[0]

        # evaluate constraint vector entry
        ck = u_bar - uc_sym 
        c_entries.append(ck)
        
    constraint_vector = csd.vertcat(*c_entries) 
    F_constraint = csd.Function('F_constraint', [uhatnp1_sym], [constraint_vector])
    return F_constraint

def get_du_matrix(mesh:TaylorImpactBar, scale):
    """method to get the constraint jacobian from analytical result
    """
    du = np.zeros((mesh.num_dofs, mesh.num_constraints))
    for k, c_node in enumerate(mesh._contact_dofs):
        du[c_node, k] = -1

    return du * scale

def compute_gaps(mesh, uhat):
    """Method to compute the x gaps between contact nodes and wall
    Inputs:
        uhat: (num_dofs x 1 numeric array) The displacement dofs for the syste
    Returns:
        gaps: (dict) mapping from contact node number to the gap value for the node
    """
    gaps = {} 
    for cnode in mesh._contact_nodes:
        X_ref = mesh.get_ref_node_position(target=cnode)
        dofs = mesh.node_dof_map[cnode].squeeze()
        u_n = uhat[np.ix_(dofs, np.array([0]))]
        x_n = X_ref + u_n 
        gaps[dofs[0]] = mesh.wall_x - x_n[0] 

    return gaps 

def create_tangent_stiffness(dt):
    # form the hessian of the residual from the unp1 parameters
    K_tan = 1/(BETA*dt**2) * M + csd.jacobian(R, uhatnp1_sym)
    F_Ktan = csd.Function('F_Ktan', [uhatnp1_sym], [K_tan])
    return F_Ktan 

# initialize velocity and acceleration 
uhatn = np.linspace(0, 0, num=2*mesh.num_nodes).reshape((2*mesh.num_nodes,1))
vhatn = []
for i in range(0, mesh.num_nodes):
    vi = np.array([[10],[0]])
    vhatn.append(vi)
vhatn = np.vstack(vhatn)
ahatn = np.linspace(0, 0, num=2*mesh.num_nodes).reshape((2*mesh.num_nodes,1))
Fnp1 = np.linspace(0, 0, num=2*mesh.num_nodes).reshape((2*mesh.num_nodes,1))
lag_mult_n = np.ones((mesh.num_constraints,1))

# plot the mesh at time tnp1 mesh... 
fnames = [] 
plot_mesh(mesh=mesh, 
            time=0,
            disp_vec=uhatn,  
            show_dof=False, 
            show_elem_num=False, 
            show_nn=False)

fn = f'gifs/01_gap/it_0.png'
fnames.append(fn)
plt.savefig(fn,dpi=300)

BETA = 0.25 
GAMMA = 0.5

uhatn_sym = csd.SX.sym('uhatn_sym', uhatnp1_sym.shape[0],1)
vhatn_sym = csd.SX.sym('vhatn_sym', uhatnp1_sym.shape[0],1)
ahatn_sym = csd.SX.sym('ahatn_sym', uhatnp1_sym.shape[0],1)
Fnp1_sym = csd.SX.sym('Fnp1_sym', uhatnp1_sym.shape[0],1)

# form residual for evaluation ... 
dt = 0.025
F_res = create_residual_map(dt=dt,
                            beta=BETA,
                            uhatn=uhatn_sym,
                            vhatn=vhatn_sym,
                            ahatn=ahatn_sym,
                            Fnp1=Fnp1_sym)

# start with a large time step to get to the wall with rigid body motion
F_Ktan = create_tangent_stiffness(dt=dt)

TOL = 10e-5
N=300
time = 0
times = [time]
# gap history over time
gap_history = dict([(cnode, []) for cnode in mesh._contact_dofs])
# save gap for history at init.
gaps = compute_gaps(mesh=mesh, 
                    uhat=uhatn)
for node, gap in gaps.items():
    gap_history[node].append(gap)
gap_violation = [(np.isclose(g,0) or g < 0) for g in gaps.values()]

lm_history = np.zeros((5,N+1))
lag_mult_sign_change = False
vel_sign_change = False
for n in range(0, N): # LOOP OVER TIME

    # initialize solution for time step
    unp1 = uhatn.copy()
    lag_mult = lag_mult_n.copy() 

    print(f'======= Time Step = {n} ======= ')
    
    if not any(gap_violation) or (lag_mult_sign_change and vel_sign_change):
        time += dt 
        times.append(time) # NOTE: THe last time step might need to be neglected for plotting?

        print('No Contact, solve standard system')
        # solve the rigid body motion until contact
        delta_u = np.zeros((uhatnp1_sym.shape[0],1))
        res_k = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full()

        it = 0
        while (np.linalg.norm(res_k) > TOL):

            # form the tangent (hessian) of the residual 
            K_k = F_Ktan(unp1).full()
            res_k = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full()

            print(f'NR-it={it}...|res_k| = {np.linalg.norm(res_k)}')
            delta_u = -np.linalg.solve(K_k, res_k)
            unp1 = unp1 + delta_u

            it += 1

        lm_history[:,n+1] = np.zeros(5)


    else: # contact has occured, solve the extended system =========
        print('Contact Detected, solve augmented system ... ')
        c = 0 
        if c == 0: # update dt and expressions
            dt = 0.0002
            # update expressions with new time
            F_Ktan = create_tangent_stiffness(dt=dt)
            F_res = create_residual_map(dt=dt,
                                        beta=BETA,
                                        uhatn=uhatn_sym,
                                        vhatn=vhatn_sym,
                                        ahatn=ahatn_sym,
                                        Fnp1=Fnp1_sym)
            F_constraint_vec = create_constraint_map(mesh=mesh)
            
            # build the extended system
            du = get_du_matrix(mesh=mesh, scale=1E6)
        time += dt
        times.append(time)
        fu = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full()

        dim = mesh.num_dofs + mesh.num_constraints
        A = scipy.sparse.csc_matrix((dim,dim))
        b = scipy.sparse.csc_matrix((dim,1))
        extended_residual = fu + du @ lag_mult
        while np.linalg.norm(extended_residual) > TOL:
                    
            # zero out the contact velocities and accelerations
            vhatn[np.ix_(mesh._contact_dofs, np.array([0]))] = 0
            ahatn[np.ix_(mesh._contact_dofs, np.array([0]))] = 0
            
            # residual
            fu = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full() 

            # constraint
            cu = F_constraint_vec(unp1).full()

            # form extended system
            A[0:mesh.num_dofs, 0:mesh.num_dofs] = F_Ktan(unp1)
            A[0:mesh.num_dofs, mesh.num_dofs::] = du 
            A[mesh.num_dofs::, 0:mesh.num_dofs] = du.T


            extended_residual = fu + du @ lag_mult
            print(f'Extended Residual = {np.linalg.norm(extended_residual)}')
            b[0:mesh.num_dofs, 0] = -1*(extended_residual)
            b[mesh.num_dofs::,0] = -1*(cu)

            # solve the extended system
            delta_sol = scipy.sparse.linalg.spsolve(A,b)
            delta_u = delta_sol[0:mesh.num_dofs].reshape((unp1.shape[0],1))
            delta_lag_mult = delta_sol[mesh.num_dofs::].reshape((lag_mult.shape[0],1))

            # update the solution and lagrange multipliers
            unp1 = unp1 + delta_u 
            lag_mult = lag_mult + delta_lag_mult

        # save lagrange multipliers for history 
        lm_history[:,n+1] = lag_mult.squeeze()

        # check for sign change in the lagrange multipliers for middle node
        if np.sign(lm_history[2,n]) != 0.0:
            if np.sign(lm_history[2,n+1]) != np.sign(lm_history[2,n]):
                lag_mult_sign_change = True

        # check sign of velocity for neighbor nodes
        if np.sign(vhatn[mesh._contact_dofs[2]-2]) < 0: # check middle node to left of contact node
            vel_sign_change = True 


    # ===== NEWMARK UPDATES for vector variables
    # update the acceleration
    ahat_np1 = (1/(BETA*dt**2)) * (unp1 - uhatn - vhatn * dt) - ((1-2*BETA)/(2*BETA)) * ahatn 

    # update the velocity 
    vhat_np1 = vhatn + ((1 - GAMMA)*ahatn + GAMMA*ahat_np1) * dt 

    # ITERATE Displacement, Vel, and Xl IN TIME
    ahatn = ahat_np1 
    vhatn = vhat_np1 
    uhatn = unp1

    # Check gaps 
    gaps = compute_gaps(mesh=mesh, 
                        uhat=unp1)
    gap_violation = [(np.isclose(g,0) or g < 0) for g in gaps.values()]

    # save gap for history
    for node, gap in gaps.items():
        gap_history[node].append(gap)

    # ====== TODO: Save and plot stress
    # TODO: back out the Second Piola Kirchoff Matrix for each element
    # TODO: plot_mesh_stress()


    # ====== TODO: Save and plot energy, kinetic and elastic 
    

    # ======= plot the mesh at time tnp1 mesh... 
    plot_mesh(mesh=mesh, 
              disp_vec=unp1,
              time=time,  
              show_dof=False, 
              show_elem_num=False, 
              show_nn=False,
              show_wire=True)
    plt.tight_layout()
    fn = f'gifs/01_gap/it_{n+1}.png'
    fnames.append(fn)
    plt.savefig(fn, dpi=300)
    plt.close()


# plot the gap and lm versus time.
fig = plt.figure()
for k, cnode in enumerate(mesh._contact_dofs):
    if k == 2:
        plt.plot(times, lm_history[k,:], '-', label=f'Node {cnode} L.M.')
        plt.plot(times, gap_history[cnode],'--', label=f'Node {cnode} Gap')
plt.xlabel('Time (s)')
plt.ylabel('Lagrange Multiplier and Gap')
plt.grid()
plt.legend()
plt.savefig(f'gifs/01_gap/lm_gap_time.png', dpi=300)


# make a gif
make_gif(gif_path='gifs/01_gap/hold_test.gif',
         filenames=fnames,
         duration=0.0002)

print('done')





