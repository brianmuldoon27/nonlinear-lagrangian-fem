"""Taylor Impact Problem using Kirchoff Saint-Venant Material Model"""

import warnings
warnings.filterwarnings("ignore")

from venv import create
import numpy as np 
import matplotlib.pyplot as plt 
import casadi as csd
import scipy 
from scipy import sparse
from scipy.sparse import linalg
from nlpfem.visual import plot_mesh, make_gif, plot_stress, plot_strain
from nlpfem.description import TaylorImpactBar

# ====== Plot Control Flags
DO_PLOT_STRESS = False 
DO_PLOT_STRAIN = False 
DO_PLOT_DEFO = False 
DO_PLOT_GAP = True 
DO_PLOT_ENERGY = True

# ======= Integration parameters
N = 200         # total number of time step
TOL = 10e-5     # Newton raphson residual tolerance

# ======== Geometric and Material parameters
# x length
LX = 1 
# y length
LY = 0.2
# number of x elems
NENX = 20
# number of y elems
NENY = 4
# material parameters
DENSITY = 10000
LAME = np.array([100E6, 40E6])
# element side length
h = LY/NENY
# bulk modulus
K = LAME[0] + 2/3*LAME[1]
# elastic modulus
E = 9*K * (K - LAME[0]) / (3*K - LAME[0])
# wave speed
c = np.sqrt(E/DENSITY)
# CFL critical time step
dt_cr = h/c 

# ======== Initialize a mesh object
mesh = TaylorImpactBar(num_elem_x=NENX,
                       num_elem_y=NENY,
                       dx=LX,
                       dy=LY,
                       density=DENSITY,
                       lame=LAME,
                       wall_x=2)

# ======= Define automatic differentiable expressions for the node dofs
# create a vertically stacked vector of displacement dof symbols for each node 
un_states = []
for n in range(0, mesh.num_nodes):
    un = csd.SX.sym(f'u_{n}',2,1)
    un_states.append(un)
uhatnp1_sym = csd.vertcat(*un_states)
uhat_ref = np.zeros((2 * mesh.num_nodes, 1))

# ======= Create automatic differentiable expressions for residual 
# mass matrix
M = mesh.mass_matrix

# stress divergence vector
R, S_avg_map, E_avg_map, J_avg_map = mesh.compute_stress_divergence_vector(uhat=uhatnp1_sym)

# ======== get the element stress tensor expression graphs
S_avg_vecs = csd.SX.zeros((3, mesh.num_elements))
for e, element in mesh.element_map.items():
    S_avg = S_avg_map[e]
    S_avg_vecs[0,e] = S_avg[0,0]
    S_avg_vecs[1,e] = S_avg[1,1]
    S_avg_vecs[2,e] = S_avg[0,1]
F_S_avg_vecs = csd.Function('F_S_avg_vecs', [uhatnp1_sym], [S_avg_vecs])

# ======== get the element strain tensor expression graphs
E_avg_vecs = csd.SX.zeros((3, mesh.num_elements))
for e, element in mesh.element_map.items():
    E_avg = E_avg_map[e]
    E_avg_vecs[0,e] = E_avg[0,0]
    E_avg_vecs[1,e] = E_avg[1,1]
    E_avg_vecs[2,e] = E_avg[0,1]
F_E_avg_vecs = csd.Function('F_E_avg_vecs', [uhatnp1_sym], [E_avg_vecs])

# ======= define a total strain energy expression graph
W_total = 0
for e, element in mesh.element_map.items():
    E_avg = E_avg_map[e]
    J_avg = J_avg_map[e]
    W_total += 4 * J_avg * (0.5 * mesh.lame[0] * (csd.trace(E_avg))**2 + mesh.lame[1] * (csd.trace(E_avg**2)))
W_map = csd.Function('W_map', [uhatnp1_sym], [W_total])

# ===== Methods to solve extended system with Newton Raphson ==== 
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
    residual = 1/(beta*dt**2) * M @ uhatnp1_sym + R - M @ ((uhatn + vhatn * dt) * (1/(beta*dt**2)) + \
                                                                     (1-2*beta)/(2*beta) * ahatn) - Fnp1
    F_res = csd.Function('F_res', [uhatnp1_sym, uhatn, vhatn, ahatn, Fnp1], [residual])
    return F_res 

def create_constraint_map(mesh):
    """Method to build the constraint vector for the system.
    Inputs:
        mesh: (TaylorImpactBar) The mesh to define constraint vector for
    Returns:
        F_constraint: (csd.Function) Mapping from unp1 (num_dofs x 1 vector) to constraint_vec (num_constraint x 1)
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
    Inputs:
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

# ====== Set up initial conditions ========== 
uhatn = np.linspace(0, 0, num=2*mesh.num_nodes).reshape((2*mesh.num_nodes,1))
vhatn = []
for i in range(0, mesh.num_nodes):
    vi = np.array([[10],[0]])
    vhatn.append(vi)
vhatn = np.vstack(vhatn)
ahatn = np.linspace(0, 0, num=2*mesh.num_nodes).reshape((2*mesh.num_nodes,1))
Fnp1 = np.linspace(0, 0, num=2*mesh.num_nodes).reshape((2*mesh.num_nodes,1))
lag_mult_n = np.ones((mesh.num_constraints,1))

# ==== plot the deformed mesh at time t=0 =======
if DO_PLOT_DEFO:
    fnames_defo = [] 
    plot_mesh(mesh=mesh, 
                time=0,
                disp_vec=uhatn,  
                show_dof=False, 
                show_elem_num=False, 
                show_nn=False)

    fn = f'gifs/03_energy/defo/it_0.png'
    fnames_defo.append(fn)
    plt.savefig(fn,dpi=300)

# ====== Plot stress at time t=0 ===============
if DO_PLOT_STRESS:
    fnames_stress = [] 
    S_avg_vecs = F_S_avg_vecs(uhatn).full()
    plot_stress(mesh=mesh,
                disp_vec=uhatn,
                show_dof=False,
                show_elem_num=False,
                show_nodes=False,
                show_nn = False,
                show_wire=True,
                S_avg_vecs=S_avg_vecs,
                time=0)
    fs_n = f'gifs/03_energy/stress/it_0.png'
    fnames_stress.append(fs_n)
    plt.savefig(fs_n,dpi=300)

# ======= Plot strain at time t = 0 ===== 
if DO_PLOT_STRAIN:
    fnames_strain = [] 
    E_avg_vecs = F_E_avg_vecs(uhatn).full()
    plot_strain(mesh=mesh,
                disp_vec=uhatn,
                show_dof=False,
                show_elem_num=False,
                show_nodes=False,
                show_nn = False,
                show_wire=True,
                E_avg_vecs=E_avg_vecs,
                time=0)
    fs_n = f'gifs/03_energy/strain/it_0.png'
    fnames_strain.append(fs_n)
    plt.savefig(fs_n,dpi=300)

# newmark integration parameters
BETA = 0.25 
GAMMA = 0.5

# autodiff expressions for evaluating residual and kinetic energy
uhatn_sym = csd.SX.sym('uhatn_sym', uhatnp1_sym.shape[0],1)
vhatn_sym = csd.SX.sym('vhatn_sym', uhatnp1_sym.shape[0],1)
ahatn_sym = csd.SX.sym('ahatn_sym', uhatnp1_sym.shape[0],1)
Fnp1_sym = csd.SX.sym('Fnp1_sym', uhatnp1_sym.shape[0],1)

# create a kinetic energy expression graph for the system
T = 0.5 * vhatn_sym.T @ M @ vhatn_sym
T_map = csd.Function('T_map', [vhatn_sym], [T])

# form residual expression graph 
dt = 0.025
F_res = create_residual_map(dt=dt,
                            beta=BETA,
                            uhatn=uhatn_sym,
                            vhatn=vhatn_sym,
                            ahatn=ahatn_sym,
                            Fnp1=Fnp1_sym)

# form tangent stiffness expression graph
F_Ktan = create_tangent_stiffness(dt=dt)

# initialize gap history container over time
gap_history = dict([(cnode, []) for cnode in mesh._contact_dofs])
# save gap for history at initiail confiug
gaps = compute_gaps(mesh=mesh, 
                    uhat=uhatn)
for node, gap in gaps.items():
    gap_history[node].append(gap)
gap_violation = [(np.isclose(g,0) or g < 0) for g in gaps.values()]

# ============== INTEGRATION ======================
time = 0            # initial time
times = [time]      # store time for plotting

# set up time history arrays
lm_history = np.zeros((5,N+1))
T_history = np.zeros(N+1)
W_history = np.zeros(N+1)
E_history = np.zeros(N+1)
T_history[0] = T_map(vhatn)
W_history[0] = W_map(uhatn)
E_history[0] = T_history[0] + W_history[0]

# set up flags for release condition 
lag_mult_sign_change = False
vel_sign_change = False

# loop over time
for n in range(0, N): 
    print(f'======= Time Step = {n} ======= ')

    # initialize solution guess for time tnp1 
    unp1 = uhatn.copy()
    lag_mult = lag_mult_n.copy() 
    
    # contact cases
    if not any(gap_violation) or (lag_mult_sign_change and vel_sign_change):
        time += dt 
        times.append(time) 

        print('-> No Contact -> Solve unconstrained system')
        # solve the rigid body motion until contact
        delta_u = np.zeros((uhatnp1_sym.shape[0],1))
        res_k = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full()

        # Newton Raphson iterations
        it = 0
        while (np.linalg.norm(res_k) > TOL):

            # form the tangent (hessian) of the residual 
            K_k = F_Ktan(unp1).full()
            res_k = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full()

            print(f'Newton Raphson -- it={it}...|residual| = {np.linalg.norm(res_k)}')
            delta_u = -np.linalg.solve(K_k, res_k)
            unp1 = unp1 + delta_u

            it += 1

        lm_history[:,n+1] = np.zeros(5)

    else: # contact has occured, solve the extended system =========
        print('-> ** Contact Detected -> solve constrained system')
        c = 0 
        if c == 0: 
            dt = 0.0002
            # update expression graphs with new time step
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
        
        # advance time    
        time += dt
        times.append(time)

        # evaluate the residual
        fu = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full()

        # form sparse containers for the extended system
        dim = mesh.num_dofs + mesh.num_constraints
        A = scipy.sparse.csc_matrix((dim,dim))
        b = scipy.sparse.csc_matrix((dim,1))
        extended_residual = fu + du @ lag_mult
        it = 0
        while np.linalg.norm(extended_residual) > TOL:
                    
            # zero out the contact dof velocities and accelerations
            vhatn[np.ix_(mesh._contact_dofs, np.array([0]))] = 0
            ahatn[np.ix_(mesh._contact_dofs, np.array([0]))] = 0
            
            # compute residual
            fu = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full() 

            # compute constraint vector
            cu = F_constraint_vec(unp1).full()

            # form extended system
            A[0:mesh.num_dofs, 0:mesh.num_dofs] = F_Ktan(unp1)
            A[0:mesh.num_dofs, mesh.num_dofs::] = du 
            A[mesh.num_dofs::, 0:mesh.num_dofs] = du.T


            extended_residual = fu + du @ lag_mult
            print(f'Newton Raphson -- it={it} -- |residual| = {np.linalg.norm(extended_residual)}')
            b[0:mesh.num_dofs, 0] = -1*(extended_residual)
            b[mesh.num_dofs::,0] = -1*(cu)

            # solve the extended system
            delta_sol = scipy.sparse.linalg.spsolve(A,b)
            delta_u = delta_sol[0:mesh.num_dofs].reshape((unp1.shape[0],1))
            delta_lag_mult = delta_sol[mesh.num_dofs::].reshape((lag_mult.shape[0],1))

            # update the solution and lagrange multipliers
            unp1 = unp1 + delta_u 
            lag_mult = lag_mult + delta_lag_mult
            it+=1

        # save lagrange multipliers for time history 
        lm_history[:,n+1] = lag_mult.squeeze()

        # check for sign change in the lagrange multipliers for middle node (release condition)
        if np.sign(lm_history[2,n]) != 0.0:
            if np.sign(lm_history[2,n+1]) != np.sign(lm_history[2,n]):
                lag_mult_sign_change = True

        # check sign of velocity for neighbor nodes (release condition)
        if np.sign(vhatn[mesh._contact_dofs[2]-2]) < 0: # check middle node to left of contact node
            vel_sign_change = True 

    # newmark update the acceleration
    ahat_np1 = (1/(BETA*dt**2)) * (unp1 - uhatn - vhatn * dt) - ((1-2*BETA)/(2*BETA)) * ahatn 

    # newmark update the velocity 
    vhat_np1 = vhatn + ((1 - GAMMA)*ahatn + GAMMA*ahat_np1) * dt 

    # update working displacement, velocity, and acceleration for next time step 
    ahatn = ahat_np1 
    vhatn = vhat_np1 
    uhatn = unp1

    # compute energies at for solution at time tnp1
    T_history[n+1] = T_map(vhatn)
    W_history[n+1] = W_map(uhatn)
    E_history[n+1] = T_history[n+1] + W_history[n+1]

    # check gaps for solution at time tnp1
    gaps = compute_gaps(mesh=mesh, 
                        uhat=unp1)
    gap_violation = [(np.isclose(g,0) or g < 0) for g in gaps.values()]

    # save gap for history
    for node, gap in gaps.items():
        gap_history[node].append(gap)

    # ======= PLOT: mesh at time tnp1 mesh... 
    if DO_PLOT_DEFO:
        plot_mesh(mesh=mesh, 
                disp_vec=unp1,
                time=time,  
                show_dof=False, 
                show_elem_num=False, 
                show_nn=False,
                show_wire=True)
        plt.tight_layout()
        fn = f'gifs/03_energy/defo/it_{n+1}.png'
        fnames_defo.append(fn)
        plt.savefig(fn, dpi=300)
        plt.close()

    # ===== PLOT: stress at time tnp1 ====
    if DO_PLOT_STRESS:
        S_avg_vecs = F_S_avg_vecs(unp1).full()
        plot_stress(mesh=mesh,
                    disp_vec=unp1,
                    show_dof=False,
                    show_elem_num=False,
                    show_nodes=False,
                    show_nn = False,
                    show_wire=True,
                    S_avg_vecs=S_avg_vecs,
                    time=time)
        fs_n = f'gifs/03_energy/stress/it_{n+1}.png'
        fnames_stress.append(fs_n)
        plt.savefig(fs_n,dpi=300)

    # ===== PLOT: strain at time tnp1 ====
    if DO_PLOT_STRAIN:
        E_avg_vecs = F_E_avg_vecs(unp1).full()
        plot_strain(mesh=mesh,
                    disp_vec=unp1,
                    show_dof=False,
                    show_elem_num=False,
                    show_nodes=False,
                    show_nn = False,
                    show_wire=True,
                    E_avg_vecs=E_avg_vecs,
                    time=time)
        fs_n = f'gifs/03_energy/strain/it_{n+1}.png'
        fnames_strain.append(fs_n)
        plt.savefig(fs_n,dpi=300)


# ===== Plot energy, kinetic and elastic 
if DO_PLOT_ENERGY:
    fig = plt.figure()
    plt.plot(times, T_history, '-r', label='kinetic energy')
    plt.plot(times, W_history, '-b', label='potential energy')
    plt.plot(times, E_history, '-g', label='total energy')

    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.grid()
    plt.legend()
    plt.savefig(f'gifs/03_energy/energy_vs_time.png', dpi=300)


# ==== plot the gap and lm versus time.
if DO_PLOT_GAP:
    fig = plt.figure()
    for k, cnode in enumerate(mesh._contact_dofs):
        if k == 2:
            plt.plot(times, lm_history[k,:], '-', label=f'$\\lambda$, Node {cnode}')
            plt.plot(times, gap_history[cnode],'--', label=f'Gap, $g$, Node {cnode} ')
        elif k == 0:
            plt.plot(times, lm_history[k,:], '-', label=f'$\\lambda$,  Node {cnode} ')
            plt.plot(times, gap_history[cnode],'--', label=f'Gap, $g$, Node {cnode} ')
    plt.xlabel('Time (s)')
    plt.ylabel('Lagrange Multiplier and Gap Function')
    plt.grid()
    plt.legend()
    plt.savefig(f'gifs/03_energy/lag_mult_gap_vs_time.png', dpi=300)

#==== make deformation gif 
if DO_PLOT_DEFO:
    make_gif(gif_path='gifs/03_energy/defo/deformation.gif',
            filenames=fnames_defo,
            duration=0.0002)

#=== make stress gif
if DO_PLOT_STRESS:
    make_gif(gif_path='gifs/03_energy/stress/stress_plots.gif',
            filenames=fnames_stress,
            duration=0.0002)

#=== make strain gif
if DO_PLOT_STRAIN:
    make_gif(gif_path='gifs/03_energy/strain/strain_plots.gif',
            filenames=fnames_strain,
            duration=0.0002)







