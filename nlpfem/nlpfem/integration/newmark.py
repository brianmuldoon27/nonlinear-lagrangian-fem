"""Class describing the implicit newmark solver"""
from typing import NoReturn
import numpy as np 
import casadi as csd 

from nlpfem.state import PhysicsState
from nlpfem.description import TaylorImpactBar

class ImplicitNewmark(object):
    """Class containing methods for the implicit newmark update.
    Inputs:
        mesh: (TaylorImpactBar) The mesh for the solver
        state: (PhysicsState) The physics state for the system at time tn
        beta: (numeric scalar) The newmark beta time stepping parameter.
        gamma: (numeric scalar) The newmark gamma time stepping parameter.
        dt: (numeric scalar) The time step for the integration.
        nr_tol: (numeric scalar) The tolerance for residual norm criterion for Newton Raphson iterations
    """
    def __init__(self, mesh:TaylorImpactBar, state:PhysicsState, beta, gamma, time_step, nr_tol=10E-5):

        self.mesh = mesh 
        """The mesh used with the solver"""

        self.state = state 
        """The physics state for the solver at time tn """

        self.beta = beta 
        """Newmark beta time stepping parameter."""

        self.gamma = gamma 
        """Newmark gamma time stepping parameter."""

        self.nr_tol = nr_tol 
        """Stopping tolernace for newton raphson iterations"""

        self._time_step = time_step
        """The time step size for the newmark method."""

        self._mass_matrix = None 
        """The mass matrix casadi expression for the system."""

        self._stress_divergence = None 
        """The stress divergence vector casadi expression for the system"""

        self._tangent_stiffness = None 
        """The tangent stiffness matrix casadi expression for the system."""

        self._residual = None 
        """The implicit newmark residual casadi expression graph for the system."""
        
        self._constraint_vector = None
        """The constraint vector casadi expression for the system."""

        self._residual_map = None 
        """Casadi Function mapping for the residual expression graph."""

        self._tangent_stiffness_map = None 
        """Casadi Function mapping for the tangent stiffness expression graph."""

        self._extended_system_map_contact = None 
        """Casadi function mapping for the extended system coefficient matrix and right hand side when in contact"""

        self._extended_system_map_free = None 
        """Casadi function mapping for the extended system coefficient matrix and right hand side when free from contact"""
        
        

    @property 
    def mass_matrix(self):
        """returns the mass matrix expression graph.
        """
        if self._mass_matrix is None:
            self._build_mass_matrix()
        return self._mass_matrix
    
    @property 
    def stress_divergence(self):
        """returns the stress divergence expression graph.
        """
        if self._stress_divergence is None:
            self._build_stress_divergence()
        return self._stress_divergence

    @property 
    def tangent_stiffness(self):
        """Returns the tangent stiffness expression graph.
        """
        if self._tangent_stiffness is None:
            self._build_tangent_stiffness()
        return self._tangent_stiffness

    @property 
    def tangent_stiffness_map(self):
        """Returns the casadi function mapping for the tangent stiffness matrix.
        """
        if self._tangent_stiffness_map is None:
            self._build_tangent_stiffness_map()
        return self._tangent_stiffness_map

    @property 
    def residual(self):
        """Returns the vectorized implicit newmark residual expression graph.
        """
        if self._residual is None:
            self._build_residual() 
        return self._residual

    @property 
    def residual_map(self):
        """Returns the casadi function mapping for the residual.
        """
        if self._residual_map is None:
            self._build_residual_map() 
        return self._residual_map

    @property 
    def constraint_vector(self):
        """Returns the vectorized symbolic constraint vector expression graph.
        """
        if self._constraint_vector is None:
            self._build_constraint_vector() 
        return self._constraint_vector

    @property 
    def extended_system_map_contact(self):
        """Returns the casadi function mapping for the extended system quantities when in contact.
        """
        if self._extended_system_map_contact is None:
            self._build_extended_system_map_contact()
        return self._extended_system_map_contact

    @property 
    def extended_system_map_free(self):
        """Returns the casadi function mapping for the extended system quantities when free.
        """
        if self._extended_system_map_free is None:
            self._build_extended_system_map_free()
        return self._extended_system_map_free

    @property 
    def time_step(self):
        """The time step for the integrator.
        """
        return self._time_step 

    @time_step.setter
    def time_step(self, dt):
        """"The setter for a time step
        """
        self._time_step = dt
    
    
    def assign_time_step(self, dt, update_maps=False):
        """Setter for the time step
        Inputs:
            dt: (numeric scalar) time step to update to
            update_maps: (bool) if True, rebuilds the mappings for residual, tangent stiffness, etc. dependent on time step
        """
        self.time_step = dt 

        if update_maps:
            # update the mappings that depend on dt 
            self._build_extended_system_map_contact()
            self._build_extended_system_map_free()

            self._build_tangent_stiffness()
            self._build_tangent_stiffness_map()

            self._build_residual()
            self._build_residual_map()



    def _build_mass_matrix(self):
        """Method to get the mass matrix expression graph for system.
        """
        self._mass_matrix = self.mesh.mass_matrix

    def _build_stress_divergence(self):
        """Method to get the stress diergence expression graph for system.
        """
        self._stress_divergence =  self.mesh.compute_stress_divergence_vector(uhat=self.state.unp1_sym)

    def _build_tangent_stiffness(self):
        """Method to build the tangent stiffness expression graph for the system.
        """
        dt = self.time_step
        beta = self.beta 
        unp1_sym = self.state.unp1_sym

        # expression graph inputs
        M = self.mass_matrix
        R = self.stress_divergence

        # assign member
        self._tangent_stiffness = 1/(beta * dt**2) * M + csd.jacobian(R, unp1_sym)

    def _build_residual(self):
        """Method to build the residual expression graph for the system.
        """
        # NOTE: This residual assumes that Fnp1 is zero!

        # unpack constants
        dt = self.time_step
        beta = self.beta

        # unpack expression graphs
        M = self.mass_matrix
        R = self.stress_divergence
        unp1 = self.state.unp1_sym
        uhatn = self.state.uhat_sym
        vhatn = self.state.vhat_sym
        ahatn = self.state.ahat_sym

        # form the differentiable residual vector
        term1 = 1/(beta*dt**2) * M @ unp1 + R 
        term2 = M @ ((uhatn + vhatn * dt) * (1/(beta*dt**2)) + (1-2*beta)/(2*beta) * ahatn) 
        
        # assign the residual attribute of the solver
        self._residual =  term1 - term2 

    def _build_constraint_vector(self):
        """Method to build the constraint vector for the system.
        """
        c_entries = []
        for c_node in self.mesh._contact_nodes:
            # get reference position for the contact node
            X_ref = self.mesh.get_ref_node_position(target=c_node)

            # get dofs associated with the contact node
            cn_dofs = self.mesh.node_dof_map[c_node]

            # get the symbol representing the contact displacement dof (x direction)
            uc_sym = self.state.unp1_sym[cn_dofs[0]]

            # get the u_bar
            u_bar = self.mesh.wall_x - X_ref[0]

            # evaluate constraint vector entry
            ck = u_bar - uc_sym 
            c_entries.append(ck)
            
        self._constraint_vector = csd.vertcat(*c_entries) 


    def _build_residual_map(self):
        """Method to build the residual casadi mapping function.
        """
        # collect input expressions
        unp1 = self.state.unp1_sym
        uhatn = self.state.uhat_sym
        vhatn = self.state.vhat_sym
        ahatn = self.state.ahat_sym

        # collect output expressions
        residual = self.residual

        # create map 
        res_map = csd.Function('res_map', [unp1, uhatn, vhatn, ahatn], 
                                          [residual], 
                                          ['unp1','uhatn', 'vhatn', 'ahatn'], 
                                          ['residual'])

        # assign map to attribute
        self._residual_map = res_map 

    def _build_tangent_stiffness_map(self):
        """method to build the tangent stiffness casadi mapping function.
        """
        # collect input expressions
        unp1 = self.state.unp1_sym

        # collect output expression
        tangent_stiffness = self.tangent_stiffness

        # build the casadi function map
        tangent_stiffness_map = csd.Function('tangent_stiffness_map', [unp1], 
                                                                      [tangent_stiffness],
                                                                      ['unp1'],
                                                                      ['tangent_stiffness'])

        # assign map to attribute
        self._tangent_stiffness_map = tangent_stiffness_map

    def _build_extended_system_map_contact(self):
        """Method to build the extended system casadi function mapping.
        """
        num_dofs = self.mesh.num_dofs 
        num_constraints = self.mesh.num_constraints

        # collect input expressions
        unp1_sym = self.state.unp1_sym
        uhatn = self.state.uhat_sym
        vhatn = self.state.vhat_sym
        ahatn = self.state.ahat_sym 
        lag_mult_sym = self.state.lag_mult_sym

        # collect output coefficient matrix expressions
        SCALE = 1E6
        du = SCALE * (csd.jacobian(self.constraint_vector, unp1_sym)).T
        Dt_du_lam = (csd.jacobian(du @ lag_mult_sym, unp1_sym)).T
        K = self.tangent_stiffness
        A_11 = K + Dt_du_lam
        A_12 = du
        A_21 = du.T 

        # assign into coefficient matrix
        dim = num_dofs + num_constraints
        A = csd.SX.zeros(dim, dim)
        A[0:num_dofs, 0:num_dofs] = A_11
        A[0:num_dofs, num_dofs::] = A_12 
        A[num_dofs::, 0:num_dofs] = A_21 

        # collect right hand side expressions
        fu = self.residual
        cu = self.constraint_vector

        # assign into right hand side vector
        du_dlam = du @ lag_mult_sym
        b1 = -1*(fu + du_dlam)
        b2 = -1*(cu)
        b = csd.vertcat(b1,b2)

        # build the casadi function map
        extended_system_map_contact  = csd.Function('extended_system_map', [unp1_sym, uhatn, vhatn, ahatn, lag_mult_sym], 
                                                                           [A, b, fu, cu, du, du_dlam, K, Dt_du_lam],
                                                                           ['u_k', 'uhatn', 'vhatn', 'ahatn', 'lag_mult_k'],
                                                                           ['A', 'b', 'fu', 'cu', 'du', 'du_dlam', 'K', 'Dt_du_lam'])

        # assign map to attribute of class
        self._extended_system_map_contact = extended_system_map_contact


    def _build_extended_system_map_free(self):
        """Method to build the extended system casadi function mapping for free system (no contact).
        """
        num_dofs = self.mesh.num_dofs 
        num_constraints = self.mesh.num_constraints

        # collect input expressions
        unp1_sym = self.state.unp1_sym
        uhatn = self.state.uhat_sym
        vhatn = self.state.vhat_sym
        ahatn = self.state.ahat_sym 
        lag_mult_sym = self.state.lag_mult_sym

        # collect output coefficient matrix expressions
        du = np.zeros((num_dofs, num_constraints))

        fu = self.residual 
        fu_eval = csd.Function('f_res_eval', [unp1_sym, uhatn, vhatn, ahatn], 
                                                [fu])
        test_val_fu = fu_eval(self.state.uhat, self.state.uhat, self.state.vhat, self.state.ahat)

        K = csd.jacobian(fu, self.state.uhat_sym)
        K_func = csd.Function('extended_system_map', [unp1_sym], 
                                                     [K])
        test_eval_K = K_func(self.state.uhat)



        A_11 = K
        A_12 = du
        A_21 = du.T 

        # assign into coefficient matrix
        dim = num_dofs + num_constraints
        A = csd.SX.zeros(dim, dim)
        A[0:num_dofs, 0:num_dofs] = A_11
        A[0:num_dofs, num_dofs::] = A_12 
        A[num_dofs::, 0:num_dofs] = A_21 

        # collect right hand side expressions
        fu = self.residual

        # assign into right hand side vector
        b1 = -1*(fu)
        b2 = np.zeros((self.mesh.num_constraints,1))
        b = csd.vertcat(b1, b2)

        # build the casadi function map
        extended_system_map_free = csd.Function('extended_system_map', [unp1_sym, uhatn, vhatn, ahatn, lag_mult_sym], 
                                                                       [A, b, fu, K],
                                                                       ['u_k', 'uhatn', 'vhatn', 'ahatn', 'lag_mult_k'],
                                                                       ['A', 'b', 'fu', 'K'])

        # assign map to attribute of class
        self._extended_system_map_free = extended_system_map_free


    def evaluate_residual(self, unp1, uhatn, vhatn, ahatn):
        """Method to evaluate the residual for the method using the symbolic maps.
        
        Inputs:
            unp1: (num_dofs x 1 numeric array) The displacement dofs at time t_np1
            uhatn: (num_dofs x 1 numeric array) The displacement dofs at time t_n
            vhatn: (num_dofs x 1 numeric array) The velocity dofs at time t_n
            ahatn: (num_dofs x 1 numeric array) The acceleration dofs at time t_n

        Returns:
            residual: (num_dofs x 1 numeric array) Numeric array of the residual vector for the system.
        """
        res = self.residual_map(unp1, uhatn, vhatn, ahatn)
        return res.full()

    def evaluate_tangent_stiffness(self, unp1):
        """Method to evaluate the tangent stiffness for a given displacement vector at instant in time.

        Inputs:
            unp1: (num_dofs x 1 numeric array) The displacement dofs at instant in time.
        
        Returns:
            tangent_stiffness: (num_dofs x num_dofs) The tangent stiffness matrix. 
        """
        tangent_stiffness = self.tangent_stiffness_map(unp1=unp1)['tangent_stiffness']
        return tangent_stiffness.full()

    def evaluate_extended_system(self, u_k, uhatn, vhatn, ahatn, lag_mult_k, in_contact):
        """Method to evaluate the extended constraint system matrix.
        
        Inputs:
            u_k: (num_dofs x 1 numeric array) The displacement dofs for iteration k of NR
            uhatn: (num_dofs x 1 numeric array) The displacement dofs for time tn
            vhatn: (num_dofs x 1 numeric array) The velocity dofs for time tn
            ahatn: (num_dofs x 1 numeric array) The acceleration dofs for time tn
            lag_mult_k: (num_constraints x 1) The lagrange multipliers for iteration k of NR
            Fnp1_k: (num_dofs x 1 ) The total force vector at time tnp1 for iteration k of NR
            in_contact: (bool) if True, the system is experience contact/penetration with wall
        
        Returns:
            A: ([num_dofs + num_constraints] x [num_dofs + num_constraints] numeric array) extended system coefficient matrix
            b: ([num_dofs + num_constraints]) numeric array) The right hand side vector for extended system.
        """
        if not in_contact:
            F = self.extended_system_map_free(u_k=u_k, 
                                              uhatn=uhatn,
                                              vhatn=vhatn,
                                              ahatn=ahatn,
                                              lag_mult_k=lag_mult_k)
        else:
            F = self.extended_system_map_contact(u_k=u_k, 
                                                 uhatn=uhatn,
                                                 vhatn=vhatn,
                                                 ahatn=ahatn,
                                                 lag_mult_k=lag_mult_k)
        
        return F


    def compute_newton_raphson(self, unp1_guess, lag_mult_guess, in_contact=False):
        """Method to perform newton raphson iterations for an instant in time to minimize residual.

        Inputs:
            unp1_guess: (num_dofs x 1 numeric array) The initial guess for the solution.
            lag_mult_guess: (num_constraints x 1 numeric array) The initial guess for the lagrange multipliers.
            in_contact: (bool) If True, solve the system using contact formulation (i.e. with non-zero lagrange multipliers)

        Returns:
            unp1: (num_dofs x 1 numeric array) The converged solution displacement dofs.
            lag_mult: (num_constraints x 1 numeric array) The converged lagrange multipliers.
        """
        # compute an initial residual 

        res = self.evaluate_residual(unp1=unp1_guess,
                                     uhatn=self.state.uhat,
                                     vhatn=self.state.vhat,
                                     ahatn=self.state.ahat)

        # initialize the NR vectors for sol.
        unp1_k = unp1_guess.copy()
        lag_mult_k = lag_mult_guess.copy()
   
        # start NR iterations
        it = 0
        residual_norm = np.linalg.norm(res)
        MAX_ITER = 10
        while residual_norm > self.nr_tol and it < MAX_ITER:
            residual_norm = np.linalg.norm(res)
            print(f'it={it}, Residual Norm = {residual_norm}')
    
            # get the extended system
            vals = self.evaluate_extended_system(u_k=unp1_k,
                                                 uhatn=self.state.uhat,
                                                 vhatn=self.state.vhat,
                                                 ahatn=self.state.ahat,
                                                 lag_mult_k=lag_mult_k,
                                                 in_contact=in_contact)
            if not in_contact:
                A, b, fu, K = vals['A'], vals['b'], vals['fu'], vals['K']
                # f-norm at time t=0 is equal to Mike
                # 4 = 4.80
            
            else:
                A, b, fu, cu, du, du_dlam, K, Dt_du_lam = (
                vals['A'], vals['b'], vals['fu'], vals['cu'], vals['du'], vals['du_dlam'], \
                vals['K'], vals['Dt_du_lam'] )
            
            # solve for the stacked delta updates
            delta_sol = csd.solve(A,b)
            delta_u = delta_sol[0:self.mesh.num_dofs].full()
            delta_lag_mult = delta_sol[self.mesh.num_dofs::].full()

            # update the lag mult and disp
            unp1_k = unp1_k + delta_u 
            lag_mult_k = lag_mult_k + delta_lag_mult

            # re-evaluate the residual
            res = self.evaluate_residual(unp1=unp1_k,
                                         uhatn=self.state.uhat,
                                         vhatn=self.state.vhat,
                                         ahatn=self.state.ahat)
            it += 1

        return unp1_k, lag_mult_k

    def beta_xl_update(self, unp1):
        """Method to perform the newmark beta update for velocity at time tnp1
        Inputs:
            unp1: (num_dofs x 1 numeric array) The displacement vector at time tnp1
        Returns:
            ahat_np1: (num_dofs x 1 numeric array) The acceleration vector at time tnp1 
        """
        # constants
        beta = self.beta 
        dt = self.time_step

        # vectors
        uhatn = self.state.uhat
        vhatn = self.state.vhat
        ahatn = self.state.ahat

        # update the acceleration
        ahat_np1 = (1/(beta*dt**2)) * (unp1 - uhatn - vhatn * dt) - ((1-2*beta)/(2*beta)) * ahatn 

        return ahat_np1


    def gamma_vel_update(self, ahat_np1):
        """Method to perform the newmark gamma upate with the velocities.
        Inputs:
            ahat_np1: (num_dofs x 1) The acceleration vector at time tnp1 
        Returns:
            vhat_np1: (num_dofs x 1) The velocity vector at time tnp1 
        """
        # constants
        gamma = self.gamma
        dt = self.time_step

        # vectors
        vhatn = self.state.vhat 
        ahatn = self.state.ahat

        # update the velocity 
        vhat_np1 = vhatn + ((1 - gamma)*ahatn + gamma*ahat_np1) * dt 

        return vhat_np1 


    def assign_displacement(self, u_new):
        """Method to update the displacement of the state for the solver.
        """
        assert u_new.shape[0] == self.state.uhat.shape[0], 'Assignment vector dimensions do not match'
        self.state.uhat = u_new 

    def assign_velocity(self, v_new):
        """Method to update the velocity dof of the state for the solver
        """
        assert v_new.shape[0] == self.state.vhat.shape[0], 'Assignment vector dimensions do not match'
        self.state.vhat = v_new 
    
    def assign_acceleration(self, a_new):
        """Method to update the acceleration dof of the state for the solver.
        """
        assert a_new.shape[0] == self.state.ahat.shape[0], 'Assignment vector dimensions do not match'
        self.state.ahat = a_new 

    def assign_lagrange_multipliers(self, lag_mult):
        """Method to update the lagrange multipliers of the state for the solver.
        """
        assert lag_mult.shape[0] == self.state.lag_mult.shape[0], 'Assignment vector dimensions do not match'
        self.state.lag_mult = lag_mult

    def compute_gaps(self, uhat):
        """Method to compute the x gaps between contact nodes and wall
        Inputs:
            uhat: (num_dofs x 1 numeric array) The displacement dofs for the syste
        Returns:
            gaps: (dict) mapping from contact node number to the gap value for the node
        """
        gaps = {} 
        for cnode in self.mesh._contact_nodes:
            X_ref = self.mesh.get_ref_node_position(target=cnode)
            dofs = self.mesh.node_dof_map[cnode].squeeze()
            u_n = uhat[np.ix_(dofs, np.array([0]))]
            x_n = X_ref + u_n 
            gaps[dofs[0]] = self.mesh.wall_x - x_n[0] 

        return gaps 

    