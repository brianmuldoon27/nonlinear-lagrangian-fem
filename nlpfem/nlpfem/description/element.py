"""Class describing a 4 node bilinear quadrilateral element."""

import numpy as np
import casadi as csd

class BilinearQuadElement(object):
    """Class representing a bilinear quadrilateral element in the natural domain.
   
    Inputs:
        num: (numeric scalar) The element number in the global mesh
        node_nums: (4 x 1 numeric array) The global node numbers associated with the element.
        node_pos: (4 x 2 symbolic array) The positions of the nodes associated with the element in (0,1,2,3) column order
        dof_nums: (2 x 4 numeric array) The displacement degree of freedom numbers for the element in (0,1,2,3) column order 
        density: (numeric scalar) the mass density of the element.
        lame: (2 x 1 numeric array) The lame parameters ordered (lambda, mu) for the element.
    """
    def __init__(self, num, node_nums, node_pos, dof_nums, density, lame):

        self._num = num
        """The element number"""

        self._node_nums = node_nums
        """The global node numbers associated with the element"""

        self._dof_nums = dof_nums
        """The global dof numbers associated with the element"""

        self._gauss_points = (-1/np.sqrt(3), 1/np.sqrt(3))
        """The gauss point locations for the element in the natural domain."""

        self._gauss_weights = (1,1)
        """The gauss weights for the gauss point in the natural domain."""

        self._node_pos = node_pos 
        """The array of reference config node positions in the global mesh."""

        self._lame = lame 
        """The lame parameters for the element."""

        self._density = density 
        """The density of the element."""

        # define the shape functions for the element
        def N1(X):
            """Interpolation function for node 1 in the natural space.
            """
            return 1/4 * (1 - X[0]) * (1 - X[1])

        def N2(X):
            """Interpolation function for node 2 in the natural space.
            """
            return 1/4 * (1 + X[0]) * (1 - X[1])

        def N3(X):
            """Interpolation funciton for node 3 in the natural space.
            """
            return 1/4 * (1 + X[0]) * (1 + X[1])

        def N4(X):
            """Interpolation function for node 4 in the natural space.
            """
            return 1/4 * (1 - X[0]) * (1 + X[1])

        self._shape_functions = {0:N1, 1:N2, 2:N3, 3:N4}
        """Mapping from local node number to shape function handles for the element."""

        # define gradients of the shape functions for the element
        def dN1_dxi(X):
            return -1/4 * (1 - X[1])

        def dN1_deta(X):
            return - 1/4 * (1 - X[0]) 

        def dN2_dxi(X):
            return 1/4 * (1 - X[1])

        def dN2_deta(X):
            return - 1/4 * (1 + X[0]) 

        def dN3_dxi(X):
            return 1/4 * (1 + X[1])
        
        def dN3_deta(X):
            return 1/4 * (1 + X[0])

        def dN4_dxi(X):
            return -1/4 * (1 + X[1])

        def dN4_deta(X):
            return 1/4 * (1 - X[0]) 

        self._shape_func_grads = {0:(dN1_dxi, dN1_deta),
                                  1:(dN2_dxi, dN2_deta),
                                  2:(dN3_dxi, dN3_deta),
                                  3:(dN4_dxi, dN4_deta)}
        """Mapping from local node number to gradients of the shape function handles associated with the node."""

    @property 
    def num(self):
        """The element number."""
        return self._num 

    @property 
    def node_nums(self):
        """The nodes numbers associated with the element in the global mesh"""
        return self._node_nums

    @property 
    def gauss_points(self):
        """Returns the tuple of gauss points for the element.
        """
        return self._gauss_points 
    
    @property 
    def gauss_weights(self):
        """Returns the tuple of gauss weights for the element.
        """
        return self._gauss_weights

    def get_shape_function(self, target):
        """Method to return the shape function handle for a desired target node.
        Inputs:
            target: (numeric int) The node number of interest for the local element. Options  include (0,1,2,3).
        Outputs:
            N: (function handle) Shape function handle for the target node
        """
        return self._shape_functions[target]

    def get_node_position(self, target):
        """Method to return a target node position vector.
        Inputs:
            target: (numeric scalar) The target node number to retreive position for.
        Returns:
            pos: (2 x 1 numeric array) The position vector associated with the target local node number in global space.
        """
        return self._node_pos[target, :].T

    def get_dof_nums(self, target):
        """Method to return the global dof numbers for a target local node number.
        Inputs:
            target: (numeric int) Target local node number to retrieve nodal dof numbers for.
        Returns:
            node_dofs: (2 x 1 array) The velocity dofs associated with the target local node
        """
        return self._dof_nums[:, target]
    
    def vectorize_dof_nums(self):
        """Method to flatten the global dof numbers for the element into an 8 x 1 array
        Returns:
            flat_dof_num: (list length 8)
        """
        flat_dof_nums = self._dof_nums.flatten(order='F')
        return flat_dof_nums
    
    def get_shape_func_grad_natural(self, target):
        """Method to return the function handle for the gradient of a target shape function with respect to a coordinate direction.
        Inputs:
            X: (2 x 1 array) Position vectors to evaluate shape function gradients at 
            target: (numeric int) The target node number of interest for shape function
        Returns:
            dN_dxi: (function handle) The function handle for the gradient of a target shape function with respect to a specified coordinate.
        """
        dNi_dxi, dNi_deta = self._shape_func_grads[target]
        return dNi_dxi, dNi_deta

    def get_shape_func_grad_physical(self, X, J_mat, target):
        """Method to return the function handle for the gradient of a target shape function with respect to a coordinate direction.
        Inputs:
            X: (2 x 1 array) Position vectors to evaluate shape function gradients at 
            target: (numeric int) The target node number of interest for shape function
        Returns:
            dN_dxi: (function handle) The function handle for the gradient of a target shape function with respect to a specified coordinate.
        """
        dNi_dxi, dNi_deta = self._shape_func_grads[target]
        dN_dxi = np.vstack((dNi_dxi(X), dNi_deta(X)))

        # take transpose of jacobian matrix 
        J_mat_T = J_mat.T

        # HARD CODE THE JACOBIAN MATRIX INVERSE
        det_JT = J_mat_T[0,0] * J_mat_T[1,1] - J_mat_T[0,1] * J_mat_T[1,0]
        adj_JT = self.make_adjoint(J=J_mat_T)
        dN_dx = adj_JT/det_JT @ dN_dxi 

        # unpack the gradient in physical space
        dN_dx1 = dN_dx[0]
        dN_dx2 = dN_dx[1]
        return dN_dx1, dN_dx2 

    def compute_N_matrix(self, X):
        """Method to compute the ( 2 x 8 numeric array) matrix of shape functions evaluated at position X
        Inputs:
            X: (2 x 1 numeric array) The position to evaluate the N matrix for.
        Returns:
            N_mat: (2 x 8 numeric array) The N matrix for the element evaluated at X.
        """
        N_mat = np.zeros((2,8))
        
        for i in range(0,4):
            Ni = self.get_shape_function(target=i)
            Ni_eval = Ni(X) * np.eye(2)
            N_mat[0:2 , 2*i:2*i+2 ] = Ni_eval 

        return N_mat 

    def compute_Bi_matrix(self, target, X):
        """Method to compute the (3 x 2 numeric array) matrix of shape function gradients for stress-div term.
        Inputs:
            target: (numeric int) The target local node to compute the Bi matrix for
            X: (2 x 1 numeric array) The position to evaluate the N matrix for.
        Returns:
            Bi: (3 x 2 numeric array) The Bi matrix for the target node
        """
        Bi = csd.SX.zeros((4,2))

        # get jacobian matrix mapping 
        J_mat = self.compute_jacobian_matrix(X=X)

        # get shape function gradients in physical space
        dN_dx1, dN_dx2 = self.get_shape_func_grad_physical(X=X, 
                                                           J_mat=J_mat,
                                                           target=target)
        # assign matrix values
        Bi[0,0] = dN_dx1
        Bi[1,1] = dN_dx2
        Bi[2,0] = dN_dx2
        Bi[3,1] = dN_dx1

        return Bi 

    def make_adjoint(self, J):
        """Creates the adjoint matrix of a 2 x 2
        Inputs:
            J: (2 x 2 symbolic array) The jacobian matrix
        Returns:
            adj_J: (2 x 2 symbolic array) the adjoint of the matrix
        """
        adj_J = csd.SX.zeros(2,2)
        adj_J[0,0] = J[1,1]
        adj_J[0,1] = -J[0,1]
        adj_J[1,0] = -J[1,0]
        adj_J[1,1] = J[0,0]
        return adj_J
        
    def compute_B_matrix(self, X):
        """Method to compute the element (3 x 8) B matrix.
        Inputs:
            X: (2 x 1 numeric array) The position vector to evaluate the B matrix 
        Returns:
            B_mat: (3 x 8 numeric array) The element B matrix
        """
        B_mat = csd.SX.zeros((4,8))
        
        for i in range(0,4):
            Bi = self.compute_Bi_matrix(target=i, X=X)
            B_mat[0:4 , 2*i:2*i+2] = Bi 

        return B_mat 

    def compute_jacobian_matrix(self, X):
        """Method to evaluate the determinant of the jacobian mapping for isoparametrics for a target X position.
        Inputs:
            X: (2 x 1 numeric array) The position to evaluate the jacobian matrix for
        Returns:
            J_matrix: (2 x 2 numeric array) The jacobian matrix evaluated at X
        """
        J_matrix = csd.SX.zeros(2,2)
        dx_dxi = csd.SX.zeros(2,1)
        dx_deta = csd.SX.zeros(2,1)
        for i in range(0,4):
            # the target node position
            xn = self.get_node_position(target=i)

            # gradients with respect to xi 
            dNi_dxi, dNi_deta = self.get_shape_func_grad_natural(target=i)
            dx_dxi  += dNi_dxi(X) * xn
            dx_deta += dNi_deta(X) * xn

        J_matrix[:,0] = dx_dxi
        J_matrix[:,1] = dx_deta

        return J_matrix

    def compute_jacobian_det(self, X):
        """Method to evaluate the determinant of the jacobian mapping for isoparametrics for a target X position.
        Inputs:
            X: (2 x 1 numeric array) The position to evaluate the jacobian determinant for
        Returns:
            J: (numeric scalar) The jacobian determinant evaluated at X
        """
        J_matrix = self.compute_jacobian_matrix(X=X)
        J_det = J_matrix[0,0]*J_matrix[1,1] - J_matrix[1,0]*J_matrix[0,1]
        return J_det

    def compute_mass_matrix(self):
        """Method to compute the element mass matrix using nodal quadrature.
        Outputs:
            M_e: (8 x 8 numeric array) 2 * NDOF X 2 * NDOF element mass matrix array
        """
        M_e = csd.SX.zeros((8,8))
        for I, gp_I in enumerate([-1,1]):
            for K, gp_K in enumerate([-1,1]):
                # define the gauss point position
                x_gp = np.array([gp_I, gp_K]).reshape((2,1))

                # evalaute the N_e matrix at the gauss point
                N_e = self.compute_N_matrix(X=x_gp)

                # evalute the Jacobian det at the gauss point
                J = self.compute_jacobian_det(X=x_gp)

                # gauss weights
                w_I = self.gauss_weights[I]
                w_J = self.gauss_weights[K]

                # compute the gaussian quadrature
                M_e_temp = self._density * N_e.T @ N_e * J * w_I * w_J
                M_e += M_e_temp
        
        return M_e


    def compute_stress_divergence_vector(self, uhat):
        """Method to compute the element stress divergence vector matrix.
        Returns:
            R_e: (8 x 1) The element stress divergence vector computed with Gaussian quadrature.
        """
        R_e = csd.SX.zeros(8,1)

        S_avg = csd.SX.zeros(2,2)
        E_avg = csd.SX.zeros(2,2)
        J_avg = 0

        for I, gp_I in enumerate(self.gauss_points):
            for K, gp_K in enumerate(self.gauss_points):
                # define the gauss point position
                x_gp = np.array([gp_I, gp_K]).reshape((2,1))

                # evalaute the B_e matrix at the gauss point
                B = self.compute_B_matrix(X=x_gp)

                # evaluate the 1st piola kirchoff
                P, S, E = self.compute_1st_piola_kirchoff(X=x_gp, u_hat=uhat) 
                S_avg += 0.25 * S 
                E_avg += 0.25 * E

                # vectorize stress
                P_vec = self.vectorize_1st_piola_kirchoff(P=P)

                # evalute the Jacobian det at the gauss point
                J = self.compute_jacobian_det(X=x_gp)
                J_avg += 0.25 * J 

                # gauss weights
                w_I = self.gauss_weights[I]
                w_J = self.gauss_weights[K]

                # NOTE: THE B MATRIX FOR THE STRESS DIVERGENCE TERM IS DIFFERENT THAN THE FLUIDS PROB
                # compute the gaussian quadrature
                R_e += w_I * w_J * B.T @ P_vec * J
        
        return R_e, S_avg, E_avg, J_avg 


    def compute_deformation_gradient(self, X, u_hat):
        """Method to evaluate the deformation gradient for a given position vector/
        Inputs:
            X: ( 2 x 1 numeric array) The position in the element to evaluate the deformation gradient.
            u_hat: (4 x 2 numeric array) The nodal degrees of freedom for the element
        Returns:
            F: (2 x 2 numeric array) the deformation gradient for a given X given in physical space
        """
        F = csd.SX.zeros((2,2))
        F[0,0] = 1
        F[1,1] = 1
        J_mat = self.compute_jacobian_matrix(X=X)
        for i in range(0,2):
            for j in range(0,2):
                for k in range(0,4):
                    dNk_dX = self.get_shape_func_grad_physical(X=X, J_mat=J_mat, target=k)
                    F[i,j] += dNk_dX[j] * u_hat[k,i]
                    
        return F 


    def compute_2nd_piola_kirchoff(self, X, u_hat):
        """Method to compute the second piola kirchoff stress tensor 

        Inputs:
            X: (2 x 1 numeric array) The position of to evaluate the stress divergence vector at
            u_hat: (4 x 2 numeric array) Array of (active) nodal degrees of freedom. Must be shape (NEN x NUMDF) = 4 x 2     
        """
        assert u_hat.shape == (4,2), f'Input u_hat must be shape (NEN x NUMDF) = 4 x 2, but got {u_hat.shape}'
        assert X.shape == (2,1), f'Input X must be shape 2 x 1, but got {X.shape}'

        # compute 2D lagrangian strain tensor 
        F = self.compute_deformation_gradient(X=X, u_hat=u_hat)
        E = 0.5 * (F.T @ F - np.eye(2))

        # compute 2nd piola kirchoff
        eye = csd.SX.zeros(2,2)
        eye[0,0] = 1
        eye[1,1] = 1
        S = self._lame[0] * csd.trace(E) * eye + 2 * self._lame[1] * E

        return S, E

    def compute_1st_piola_kirchoff(self, X, u_hat):
        """Method to compute the first piola kirchoff stress tensor 
        Inputs:
            X: (2 x 1 numeric array) The position of to evaluate the stress divergence vector at
            u_hat: (4 x 2 numeric array) Array of (active) nodal degrees of freedom. Must be shape (NEN x NUMDF) = 4 x 2     
        """
        assert u_hat.shape == (4,2), f'Input u_hat must be shape (NEN x NUMDF) = 4 x 2, but got {u_hat.shape}'
        assert X.shape == (2,1), f'Input X must be shape 2 x 1, but got {X.shape}'

        F = self.compute_deformation_gradient(X=X, u_hat=u_hat)
        S, E = self.compute_2nd_piola_kirchoff(X=X, u_hat=u_hat)
        P = F @ S

        return P, S, E

    def vectorize_1st_piola_kirchoff(self, P):
        """Method to vectorize the 1st piola kirchoff tensor components.
        Inputs:
            P: (2 x 2 numeric array) The 1st piola kirchoff stress tensor
        Oupts:
            P_vec: (4 x 1 numeric array) The vectorized 1st piola kirchoff
        """
        elems = (P[0,0], P[1,1], P[0,1], P[1,0])
        P_vec = csd.vertcat(*elems)
        return P_vec



