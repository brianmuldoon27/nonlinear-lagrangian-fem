"""Supporting files for mesh generation of the cavity problem."""

import numpy as np
import casadi as csd 
from nlpfem.description import BilinearQuadElement

class TaylorImpactBar(object):
    """Class representing the mesh information for the taylor impact problem.

    Inputs:
        num_elem_x: (numeric scalar) The number of elements along the horizontal direciton.
        num_elem_y: (numeric scalar) The number of elements along the vertical dimension.
        dx: (numeric scalar) The side length (x) for the bar 
        dy: (numeric scalar) The side length (y) for the bar
        density: (numeric scalar) The mass density of the material
        lame: (2 x 1 array) The lame constants for the material ordered (lambda, mu)
        wall_x: (numeric scalar) The x location of the impact wall 
    """
    def __init__(self, num_elem_x, num_elem_y, dx, dy,  density, lame, wall_x):

        self.num_elem_x = num_elem_x
        """Number of row elements for the mesh """

        self.num_elem_y = num_elem_y
        """Number of col elements for the mesh """

        self.dy = dy
        """The vertical dimension (y) of the bar"""

        self.dx = dx
        """The horizontal dimension (x) of the bar"""

        self.density = density
        """The density of the material"""

        self.lame = lame
        """The lame parameters for the material."""

        self.wall_x = wall_x
        """The x location of the impact wall"""

        self.elem_side_dx = self.dx/self.num_elem_x
        """The horizontal side length of an element."""
    
        self.elem_side_dy = self.dy/self.num_elem_y
        """The vertiacl side length of an element."""

        self.num_elements = num_elem_x * num_elem_y
        """The total number of elements in the mesh"""

        self.num_nodes_x_side = self.num_elem_x + 1
        """The total number of nodes along the horizontal dimension"""

        self.num_nodes_y_side = self.num_elem_y + 1
        """The total number of nodes along the horizontal dimension"""

        self.num_nodes = self.num_nodes_y_side * self.num_nodes_x_side
        """The total number of nodes in the mesh."""

        self.nodes = tuple(np.arange(0,self.num_nodes))
        """Tuple of all node numbers in the mesh """

        self._mass_matrix = None 
        """The global mass matrix for the mesh."""

        self._stress_divergence_vector = None
        """The global stress divergence vector for the mesh. """

        self._element_map = {}
        """Mapping from element number to element object."""

        self._ref_node_positions = {}
        """Mapping from node number to reference node position vector """

        self._node_dof_map = {}
        """Mapping from node number to associated dof number 2 x 1 np.array """

        # Create node numbering and dof allocation 
        node_num = 0
        contact_nodes = [] 
        active_nodes = []
        for r in range(0, self.num_nodes_y_side):
            # compute the y coordinate of the node
            yn = r * self.elem_side_dy

            for c in range(0, self.num_nodes_x_side):
                # compute the x coordinate of the node
                xn = c * self.elem_side_dx

                # add node position to global node position array
                self._ref_node_positions[node_num] = np.vstack((xn,yn)).reshape((2,1))

                # check if it is a right edge node
                if np.isclose(xn, self.dx):
                    contact_nodes.append(node_num)
                else:
                    active_nodes.append(node_num)
                node_num += 1

        self._contact_nodes = np.vstack(contact_nodes).squeeze()
        self._active_nodes = np.vstack(active_nodes).squeeze()

        # instantiate the elements in the mesh and assign the nodal positions for the nodes
        elem_count = 0
        for r in range(0, self.num_elem_y):
            for c in range(0, self.num_elem_x):
                # assign node numbers to the respective elements
                node_1_num = r * self.num_nodes_x_side + c 
                node_2_num = r * self.num_nodes_x_side + 1 + c 
                node_3_num = (r + 1) * self.num_nodes_x_side + 1 + c
                node_4_num = (r + 1) * self.num_nodes_x_side + c
                node_nums = (node_1_num, node_2_num, node_3_num, node_4_num)

                # define the dof numbers per node
                n1_dof = np.array([[2*node_1_num], [2*node_1_num+1]])
                n2_dof = np.array([[2*node_2_num], [2*node_2_num+1]])
                n3_dof = np.array([[2*node_3_num], [2*node_3_num+1]])
                n4_dof = np.array([[2*node_4_num], [2*node_4_num+1]])
                dof_nums = np.hstack((n1_dof, n2_dof, n3_dof, n4_dof))

                # create mapping from node number to velocity dofs
                self._node_dof_map[node_1_num] = n1_dof
                self._node_dof_map[node_2_num] = n2_dof 
                self._node_dof_map[node_3_num] = n3_dof 
                self._node_dof_map[node_4_num] = n4_dof 

                # get the positions of the nodes for the element
                p1 = self._ref_node_positions[node_1_num]
                p2 = self._ref_node_positions[node_2_num]
                p3 = self._ref_node_positions[node_3_num]
                p4 = self._ref_node_positions[node_4_num]
                node_pos = np.hstack((p1,p2,p3,p4)).T

                # create an element object
                elem = BilinearQuadElement(num=elem_count,
                                           node_nums=node_nums,
                                           node_pos=node_pos,
                                           dof_nums=dof_nums,
                                           density=self.density,
                                           lame=self.lame)

                # add the element to the mesh
                self._element_map[elem_count] = elem

                # iterate the element counter
                elem_count += 1

        # define the contact dof nums
        contact_dofs = []
        active_dofs = [] 
        for nn in self._contact_nodes:
            cdof = self.node_dof_map[nn]
            contact_dofs.append(cdof[0]) # save only the 1 direction as a contact dof
            active_dofs.append(cdof[1])  # save the 2 direction of the contact node to active dofs

        self._contact_dofs = np.vstack(contact_dofs).squeeze()

        # define the active dof nums
        for nn in self._active_nodes:
            adof = self.node_dof_map[nn]
            active_dofs.append(adof)
        self._active_dofs = np.vstack(active_dofs).squeeze()

    @property 
    def num_dofs(self):
        """Returns the number of degrees of freedom in the mesh.
        """
        return 2 * self.num_nodes
    
    @property 
    def num_constraints(self):
        """Returns the number of constraint nodes in the mesh.
        """
        return self._contact_dofs.shape[0]

    @property 
    def active_elements(self):
        """Returns a list of the active element numbers
        """
        active_elem = list(np.arange(0, self.num_elements))
        active_elem.remove(self._prescribed_pressure_elem)
        return active_elem

    @property
    def ref_node_positions(self):
        """Returns the mapping from node number to 2 x 1 nodal position vector 
        Returns:
            node_positions: (dict) mapping from node number to 2 x 1 nodal position vector 
        """
        return self._ref_node_positions

    def get_ref_node_position(self, target):
        """Returns the reference position of a target node
        Inputs:
            target: (int) the target node number to get posiitn for.
        returns:
            position: (2 x 1 numeric array) The reference position coordinates for the node.
        """
        return self.ref_node_positions[target]

    @property 
    def node_dof_map(self):
        """Returns the mapping from node number to the velocity dof.
        Returns:
            node_vdof_map: (dict) mapping from node number to (2,) np.array of vdof numbers
        """
        return self._node_dof_map

    @property 
    def element_map(self):
        """Returns mapping from element number to element object.
        Returns:
            element_map: (dict) Mapping from element number to BilinearQuadElement instance
        """
        return self._element_map

    @property 
    def mass_matrix(self):
        """Returns the global mass matrix for the mesh.
        Returns:
            M_global: (2 * num_nodes x 2*num_nodes) Global mass matrix for the mesh.
        """
        if self._mass_matrix is None:
            self._build_mass_matrix() 
        return self._mass_matrix

    @property 
    def stress_divergence_vector(self):
        """Returns the global stress divergence vector the mesh.
        Returns:
            R_global: (2 * num_nodes x 1) Global stress divergence vector for the mesh
        """
        if self._stress_divergence_vector is None:
            self._build_stress_divergence_vector()
        return self._stress_divergence_vector

    def _build_mass_matrix(self):
        """Builds the global mass matrix for the mesh."""
        self._mass_matrix = self.compute_global_mass_matrix()

    def _build_stiffness_matrix(self):
        """Builds the global stiffness matrix for the mesh."""
        self._stiffness_matrix = self.compute_global_stiffness_matrix() 

    def _build_stress_divergence_vector(self):
        """Builds the global stress divergence vector for the mesh.
        """
        self._stress_divergence_vector = self.compute_stress_divergence_vector()

    def get_element(self, target):
        """Method to return an element object give target element number.
        Inputs:
            target: (numeric int) The target element number to retrieve 
        Returns:
            element: (BilinearQuadElement) The target instance element within the mesh 
        """
        return self._element_map[target]

    def compute_global_mass_matrix(self):
        """Method to compute the global mass matrix for the mesh. Assumes same element mass matrices.
        Returns:
            M_global: (2 * num_nodes x 2*num_nodes) The global mass matrix for the system
        """
        elem = self.get_element(target=0)
        M_e = elem.compute_mass_matrix()   
        M_global = csd.SX.zeros((2*self.num_nodes, 2*self.num_nodes))

        for e, element in self.element_map.items():
            dof_ind = element.vectorize_dof_nums()
            M_global[np.ix_(dof_ind, dof_ind)] += M_e 
        
        return M_global

    def compute_stress_divergence_vector(self, uhat):
        """Method to compute the global stress divergence vector for a given global displacement vector.
        """
        R_global = csd.SX.zeros((2*self.num_nodes, 1))
        R_elems = [] 
        S_avg_map = {}
        E_avg_map = {} 
        J_avg_map = {} 
        for e, element in self.element_map.items():
            dof_ind = element.vectorize_dof_nums()
            uhat_e = uhat[dof_ind].reshape((2,4)).T
            R_e, S_avg, E_avg, J_avg = element.compute_stress_divergence_vector(uhat=uhat_e)
            S_avg_map[e] = S_avg 
            E_avg_map[e] = E_avg 
            J_avg_map[e] = J_avg 

            R_elems.append(R_e)
            R_global[np.ix_(dof_ind, np.array([0]))] += R_e 

        return R_global, S_avg_map, E_avg_map, J_avg_map


