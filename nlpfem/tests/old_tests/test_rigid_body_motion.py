"""TEst to build meshes and check values"""
from xmlrpc.server import resolve_dotted_attribute
import numpy as np 
import matplotlib.pyplot as plt 
import casadi as csd
from nlpfem.visual import plot_mesh, make_gif
from nlpfem.description import TaylorImpactBar
import time 

# problem set up
# x length
LX = 1 

# y length
LY = 0.2

# number of x elems
NENX = 8

# number of y elems
NENY = 3

DENSITY = 10000

LAME = np.array([100E6, 40E6])

# make a mesh
mesh = TaylorImpactBar(num_elem_x=NENX,
                       num_elem_y=NENY,
                       dx=LX,
                       dy=LY,
                       density=DENSITY,
                       lame=LAME)

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

dt = 0.025
BETA = 0.25 
GAMMA = 0.5

fnames = [] 
# plot the mesh at time tnp1 mesh... 
plot_mesh(mesh=mesh, 
            disp_vec=uhatn, 
            wall_x=LX+1, 
            show_dof=False, 
            show_elem_num=False, 
            show_nn=False)

fn = f'gifs/rigid_motion/it_0.png'
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

DU_TOL = 10e-4
for i in range(0, 4): # LOOP OVER TIME
    print(f'======= Time Step = {i} ======= ')

    # ========= INNER LOOP FOR NR ITERATIONS  =========
    delta_u = np.ones((uhatnp1_sym.shape[0],1))
    it = 0
    unp1 = uhatn.copy()
    Fnp1 = np.zeros((uhatnp1_sym.shape[0],1))
    while np.linalg.norm(delta_u) > DU_TOL:
        # solve for delta_u_k
        K_tan_k = F_Ktan(uhatn).full()
        res_k = F_res(unp1, uhatn, vhatn, ahatn, Fnp1).full()
        print(f'NR-it={it}...|res_k| = {np.linalg.norm(res_k)}')
        delta_u = -np.linalg.solve(K_tan_k, res_k)
        unp1 += delta_u 
        it += 1

    # update the acceleration
    ahat_np1 = (1/(BETA*dt**2)) * (unp1 - uhatn - vhatn * dt) - ((1-2*BETA)/(2*BETA)) * ahatn 

    # update the velocity 
    vhat_np1 = vhatn + ((1 - GAMMA)*ahatn + GAMMA*ahat_np1) * dt 

    # plot the mesh at time tnp1 mesh... 
    plot_mesh(mesh=mesh, 
              disp_vec=unp1, 
              wall_x=LX+1, 
              show_dof=False, 
              show_elem_num=False, show_nn=False)

    fn = f'gifs/rigid_motion/it_{i+1}.png'
    fnames.append(fn)
    plt.savefig(fn,dpi=300)

    # NOTE: Mesh plotting is not drawing lines between nodes, instead just flat lines .... need to fix this... 

    # ITERATE IN TIME
    ahatn = ahat_np1 
    vhatn = vhat_np1 
    uhatn = unp1 

# make a gif
make_gif(gif_path='gifs/rigid_motion/rig_motion.gif',filenames=fnames)

print('done')





