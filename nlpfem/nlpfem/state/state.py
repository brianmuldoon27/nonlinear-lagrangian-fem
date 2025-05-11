"""Class describing the physics state of the mesh.
"""
import numpy as np 
import casadi as csd 
from nlpfem.description import TaylorImpactBar

class PhysicsState(object):
    """Class describing the physics state of a given mesh.

    Inputs:
        mesh: (TaylorImpactBar) The mesh to generate a physics state instance for.
        unp1_sym: (2 * num_nodes x 1 symbolic array) Global vector of casadi symbolic displacement dofs to solve for at time tnp1 
        uhat_sym: (2*num_nodes x 1 symbolic array) Global vector of casadi symbolic displacement degrees of freedom.
        vhat_sym: (2*num_nodes x 1 symbolic array) Global vector of casadi symbolic velocity degrees of freedom.
        ahat_sym: (2*num_nodes x 1 symbolic array) Global vector of casadi symbolic acceleration degrees of freedom.
        lag_mult_sym: (num_constraints x 1 symbolic array) Global vector of casadi symbolic lagrange multipliers.
        uhat: (2 * num_nodes x 1 numeric array) Global vector of displacement degrees of freedom at an instant in time.
        vhat: (2 * num_nodes x 1 numeric array) Global vector of nodal dof velocities at an instant in time.
        ahat: (2 * num_nodes x 1 numeric array) Global vector of nodal dof accelerations at an instant in time.
        lag_mult: (num_constraints x 1 numeric array) Global vector for lagrange multipliers at an instant in time.
    """
    def __init__(self, mesh:TaylorImpactBar, unp1_sym=None, uhat_sym=None, vhat_sym=None, ahat_sym=None, lag_mult_sym=None,
                                                            uhat=None,     vhat=None,     ahat=None,     lag_mult=None):

        self.mesh = mesh 
        """The mesh used to construct a physics state from. """

        if unp1_sym is None:
            unp1_states = []
            for n in range(0, mesh.num_nodes):
                unp1_n = csd.SX.sym(f'unp1_{n}',2,1)
                unp1_states.append(unp1_n)
            unp1_sym = csd.vertcat(*unp1_states)
        self._unp1_sym = unp1_sym
        """Vector of casadi symbolic displacement dofs at time t_np1"""

        if uhat_sym is None:
            un_states = []
            for n in range(0, mesh.num_nodes):
                un = csd.SX.sym(f'u_{n}',2,1)
                un_states.append(un)
            uhat_sym = csd.vertcat(*un_states)
        self._uhat_sym = uhat_sym
        """Vector of casadi symbolic displacement dofs at time tn."""

        if vhat_sym is None:
            vn_states = []
            for n in range(0, mesh.num_nodes):
                vn = csd.SX.sym(f'v_{n}',2,1)
                vn_states.append(vn)
            vhat_sym = csd.vertcat(*vn_states)
        self._vhat_sym = vhat_sym 
        """Vector of casadi symbolic velocity dofs at time tn ."""

        if ahat_sym is None:
            an_states = []
            for n in range(0, mesh.num_nodes):
                an = csd.SX.sym(f'a_{n}',2,1)
                an_states.append(an)
            ahat_sym = csd.vertcat(*an_states)
        self._ahat_sym = ahat_sym 
        """Vector of casadi symbolic acceleration dofs at time tn."""

        if lag_mult_sym is None:
            lm_states = []
            for n in range(0, mesh.num_constraints):
                lm_n = csd.SX.sym(f'lagmult_{n}',1)
                lm_states.append(lm_n)
            lag_mult_sym = csd.vertcat(*lm_states)
        self._lag_mult_sym = lag_mult_sym 
        """Vector of casadi symbolic acceleration dofs at time tn."""

        if uhat is None:
            uhat = np.zeros((mesh.num_dofs,1))
        self._uhat = uhat 
        """Vector of displacement dofs at time tn."""

        if vhat is None:
            vhat = np.zeros((mesh.num_dofs,1))
        self._vhat = vhat 
        """Vector of velocity dofs at time tn."""

        if ahat is None:
            ahat = np.zeros((mesh.num_dofs,1))
        self._ahat = ahat 
        """Vector of acceleration dofs at time tn."""

        if lag_mult is None:
            lag_mult = np.zeros((mesh.num_constraints,1))
        self._lag_mult = lag_mult 
        """Vector of lagrange multipliers at time tn."""

    @property 
    def unp1_sym(self):
        """The vectorized symbolic displacement dofs at time tnp1.
        Returns:
            unp1_sym: (2 * num_nodes x 1 symbolic array) Vectorized displacement dymbolic dof values for the mesh at time tnp1.
        """
        return self._unp1_sym

    @property 
    def uhat_sym(self):
        """The vectorized symbolic displacement dofs at time tn.
        Returns:
            uhat_sym: (2 * num_nodes x 1 symbolic array) Vectorized displacement dymbolic dof values for the mesh.
        """
        return self._uhat_sym

    @property 
    def vhat_sym(self):
        """The vectorized symbolic velocity dofs at time tn .
        Returns:
            vhat_sym: (2 * num_nodes x 1 symbolic array) Vectorized velocity symbolic dofs for the mesh.
        """
        return self._vhat_sym

    @property 
    def ahat_sym(self):
        """The vectorized symbolic acceleration dofs at time tn.
        Returns:
            ahat_sym: (2 * num_nodes x 1 symbolic array) Vectorized acceleration symbolic dofs for the mesh.
        """
        return self._ahat_sym

    @property 
    def lag_mult_sym(self):
        """The vectorized symbolic lagrange multipliers.
        Returns:
            ahat_sym: (num_constraints x 1 symbolic array) Vectorized lagrange multiplier symbolic dofs for the mesh.
        """
        return self._lag_mult_sym
    
    @property 
    def uhat(self):
        """The vectorized displacement state at an instant in time.
        Returns:
            uhat: (2 * num_nodes x 1 numeric array) Vectorized displacement dof values for the mesh.
        """
        return self._uhat 

    @uhat.setter
    def uhat(self, u):
        """Setter for the displacement uhat vector.
        """
        self._uhat = u

    @property 
    def vhat(self):
        """The vectorized velocity state at an instant in time.
        Returns:
            vhat: (2 * num_nodes x 1 numeric array) Vectorized velocity dof values for the mesh.
        """
        return self._vhat 

    @vhat.setter
    def vhat(self, v):
        """Setter for the velocity vhat vector.
        """
        self._vhat = v

    @property 
    def ahat(self):
        """The vectorized acceleration state at an instant in time.
        Returns:
            ahat: (2 * num_nodes x 1 numeric array) Vectorized acceleration dof values for the mesh.
        """
        return self._ahat 

    @ahat.setter
    def ahat(self, a):
        """Setter for the acceleration ahat vector.
        """
        self._ahat = a

    @property 
    def lag_mult(self):
        """The vectorized lagrange multipliers at an instant in time.
        Returns:
            lag_mult: (num_constraints x 1 numeric array) Vectorized lagrange multipliers for the mesh.
        """
        return self._lag_mult
    
    @lag_mult.setter
    def lag_mult(self, l):
        """Setter for the lagrange multiplier vector.
        """
        self._lag_mult = l




