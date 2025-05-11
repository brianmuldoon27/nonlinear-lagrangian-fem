"""Test to build meshes and check values"""
from xmlrpc.server import resolve_dotted_attribute
import numpy as np 
import matplotlib.pyplot as plt 
import casadi as csd
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

def residual(dt, beta, uhatn, vhatn, ahatn, Fnp1):
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
    return residual 

# ===== TEST Newmark TIME STEPPING ====

# initialize velocity and acceleration 
uhatn = np.linspace(0, 0, num=2*mesh.num_nodes).reshape((2*mesh.num_nodes,1))
vhatn = []
for i in range(0, mesh.num_nodes):
    vi = np.array([[10],[0]])
    vhatn.append(vi)
vhatn = np.vstack(vhatn)
ahatn = np.linspace(0, 0, num=2*mesh.num_nodes).reshape((2*mesh.num_nodes,1))
Fnp1 = np.linspace(0, 0, num=2*mesh.num_nodes).reshape((2*mesh.num_nodes,1))

dt = 0.0002
BETA = 0.25 
GAMMA = 0.5

fnames = [] 
# plot the mesh at time tnp1 mesh... 
plot_mesh(mesh=mesh, 
            time=0,
            disp_vec=uhatn,  
            show_dof=True, 
            show_elem_num=False, 
            show_nn=True)

fn = f'gifs/hold/it_0.png'
fnames.append(fn)
plt.savefig(fn,dpi=300)

# form the hessian of the residual from the unp1 parameters
K_tan = 1/(BETA*dt**2) * M + csd.jacobian(R, uhatnp1_sym)
F_Ktan = csd.Function('F_Ktan', [uhatnp1_sym], [K_tan])

uhatn_sym = csd.SX.sym('uhatn_sym', uhatnp1_sym.shape[0],1)
vhatn_sym = csd.SX.sym('vhatn_sym', uhatnp1_sym.shape[0],1)
ahatn_sym = csd.SX.sym('ahatn_sym', uhatnp1_sym.shape[0],1)
Fnp1_sym = csd.SX.sym('Fnp1_sym', uhatnp1_sym.shape[0],1)

# form residual for evaluation ... 
res_sym = residual(dt=dt,
                    beta=BETA,
                    uhatn=uhatn_sym,
                    vhatn=vhatn_sym,
                    ahatn=ahatn_sym,
                    Fnp1=Fnp1_sym)
# casadi function mapping for residual 
F_res = csd.Function('F_res', [uhatnp1_sym, uhatn_sym, vhatn_sym, ahatn_sym, Fnp1_sym], [res_sym])

TOL = 10e-5
t = 0
for i in range(0, 100): # LOOP OVER TIME
    print(f'======= Time Step = {i} ======= ')
    t += (i+1) * dt 

    # ========= INNER LOOP FOR NR ITERATIONS  =========
    delta_u = np.ones((uhatnp1_sym.shape[0],1))
    it = 0
    unp1 = uhatn.copy()
    res_k = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full()
    res_active_k = res_k[np.ix_(mesh._active_dofs, np.array([0]))]
    unp1_active = unp1[np.ix_(mesh._active_dofs,np.array([0]))]
    while (np.linalg.norm(res_active_k) > TOL):

        # enforce boundary conditions at the right end
        cind = mesh._contact_dofs
        aind = mesh._active_dofs
        numcn = mesh._contact_dofs.shape[0]

        # form the tangent (hessian) of the residual 
        K_tan_k = F_Ktan(unp1).full()
        res_k = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full()

        # ==== reduce to active system ====
        # unpack the active parts of the linear system
        K_11 = K_tan_k[np.ix_(aind, aind)] 
        res_active_k = res_k[np.ix_(mesh._active_dofs, np.array([0]))]

        print(f'NR-it={it}...|res_k| = {np.linalg.norm(res_active_k)}')
        delta_u = -np.linalg.solve(K_11, res_active_k)
        unp1_active = unp1_active + delta_u

        unp1[np.ix_(aind, np.array([0]))] = unp1_active
        
        it += 1

    # update the acceleration
    ahat_np1 = (1/(BETA*dt**2)) * (unp1 - uhatn - vhatn * dt) - ((1-2*BETA)/(2*BETA)) * ahatn 

    # update the velocity 
    vhat_np1 = vhatn + ((1 - GAMMA)*ahatn + GAMMA*ahat_np1) * dt 

    # plot the mesh at time tnp1 mesh... 
    plot_mesh(mesh=mesh, 
              disp_vec=unp1,
              time=t,  
              show_dof=False, 
              show_elem_num=False, 
              show_nn=False,
              show_wire=True)
    
    # TODO: back out the Second Piola Kirchoff Matrix for each element
    # TODO: plot_mesh_stress()

    fn = f'gifs/hold/it_{i+1}.png'
    fnames.append(fn)
    plt.savefig(fn,dpi=300)

    # ITERATE IN TIME
    ahatn = ahat_np1 
    vhatn = vhat_np1 
    uhatn = unp1 

# make a gif
make_gif(gif_path='gifs/hold/hold_test.gif',
         filenames=fnames,
         duration=0.0002)

print('done')





