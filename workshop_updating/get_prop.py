# Import necessary packages
import numpy as np
import scipy.integrate as scp


def get_prop(E = 210e9):
    """Function that calculates the eigenfrequencies and eigenmodes of the beam 
    given the Young's modulus E. The Young's modulus is the parameter to optimise 
    and other material and structural properties are fixed.

    Input
    ----------
    E: Young modulus [N/m^2]

    Returns
    -------
    omega: Eigenfrequencies [rad/s] (40, )
    phi: Eigenmodes (in columns) (40, 40)

    Both as numpy array
    """

    # Define properties of beam 
    b = 0.050             # [m]
    h = 0.025             # [m]
    L = 1                 # [m]
    rho = 7750            # [kg/m^3]
    A = b*h               # [m^2]
    I = (1/12)*b*h**3       # [m^4]

    # Total number of elements
    n_elements = 20

    # Total number of nodes, nn = ne + 1
    n_nodes = n_elements + 1

    # Total number of degrees of freedom, 2 per node, 1 for displacement and 1 for rotation
    n_dofs = 2*n_nodes

    # Create empty lists for the elemental degrees of freedom, so we can specify which nodes are connected to each element
    elem_dofs = []
    # Create empty lists for the nodeall degrees of freedom, so we can specify what degrees of freedom each node has
    dof_node = []

    # Loop over the elements and append the the elementall degrees of freedom
    for ie in np.arange(0, n_elements):
        elem_dofs.append(np.arange(2*ie, 2*ie + 4))

    # Loop over nodes and append the specific nodeall degrees of freedom to each node   
    for idof in np.arange(0, n_dofs):
        dof_node.append(int(np.floor(idof/2)))

    # Creating Shape functions, probably want to put this in a seperate python file
    N_k = []
    dN_k = []
    ddN_k = []
    h = L/n_elements # = 1/20 = 0.05
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, h,  h**2, h**3], [0, 1, 2*h, 3*h**2]])
    
    # These functions will be used to obtain the local mass and stiffness matrix later
    def base(x):
        if isinstance(x, float):
            return np.array([[1], [x], [x**2], [x**3]])
        else:
            return np.array([[np.ones(len(x))], [x], [x**2], [x**3]])
    def dbase(x):
        if isinstance(x, float):
            return np.array([[0], [1], [2*x], [3*x**2]])
        else:
            return np.array([[np.zeros(len(x))], [np.ones(len(x))], [2*x], [3*x**2]])
    def ddbase(x):
        if isinstance(x, float):
            return np.array([[0], [0], [2], [6*x]])
        else:
            return np.array([[np.zeros(len(x))], [np.zeros(len(x))], [2*np.ones(len(x))], [6*x]])
    def make_N(coeff): 
        return lambda x: np.dot(np.transpose(base(x)), coeff)
    def make_dN(coeff):
        return lambda x: np.dot(np.transpose(dbase(x)), coeff)
    def make_ddN(coeff):
        return lambda x: np.dot(np.transpose(ddbase(x)), coeff)


    dof_vec = np.arange(0,4)

    for idof in dof_vec:
        rhs = np.zeros(len(dof_vec))
        rhs[idof] = 1
        coeff = np.linalg.solve(matrix, rhs)
        N_k.append(make_N(coeff))
        dN_k.append(make_dN(coeff))
        ddN_k.append(make_ddN(coeff))

    M_k = np.zeros((4, 4))
    K_k = np.zeros((4, 4))
    for idof in np.arange(0, 4):
        for jdof in np.arange(0, 4):
            def eqn_M(x):
                return rho*A*N_k[idof](x)*N_k[jdof](x)
            def eqn_K(x): 
                return E*I*ddN_k[idof](x)*ddN_k[jdof](x)
            M_k[idof, jdof] = scp.quad(eqn_M, 0, h)[0]
            K_k[idof, jdof] = scp.quad(eqn_K, 0, h)[0]

    K = np.zeros((n_dofs*n_dofs))       # 1-D array for global stiffness matrix
    M = np.zeros((n_dofs*n_dofs))       # 1-D array for global mass matrix


    for ie in np.arange(0, n_elements):
        # Get the nodes of the elements
        dofs = elem_dofs[ie]
        NodeLeft = dof_node[dofs[0]]
        NodeRight = dof_node[dofs[-1]]
        
        # Get the degrees of freedom that correspond to each node
        Dofs_Left = 2*(NodeLeft) + np.arange(0, 2)
        Dofs_Right = 2*(NodeRight) + np.arange(0, 2)

        # Assemble the matrices at the correct place
        nodes = np.append(Dofs_Left, Dofs_Right)
        for i in np.arange(0, 4):
            # Q[nodes[i]] = Q[nodes[i]] + Q2_k[ie][i](0.01)
            for j in np.arange(0, 4):
                ij = nodes[j] + nodes[i]*n_dofs
                M[ij] = M[ij] + M_k[i, j]
                K[ij] = K[ij] + K_k[i, j]
                
    # Reshape the global matrix from a 1-D array to a 2-D array
    M = M.reshape((n_dofs, n_dofs))
    K = K.reshape((n_dofs, n_dofs))

    fixed_dofs = np.arange(0, 2)                    # fixed DOFs
    free_dofs = np.arange(0, n_dofs)                # free DOFs
    free_dofs = np.delete(free_dofs, fixed_dofs)    # remove the fixed DOFs from the free DOFs array

    # free & fixed array indices
    fx = free_dofs[:, np.newaxis]
    fy = free_dofs[np.newaxis, :]

    # Mass
    Mii = M[fx, fy]

    # Stiffness
    Kii = K[fx, fy]

    # calculating the eigenvalues and right eigenvectors
    mat = np.dot(np.linalg.inv(Mii), Kii)
    omega2, vr = np.linalg.eig(mat)
    omega = np.sqrt(omega2)
    omega_sort_index = np.argsort(omega)
    omega_sort = np.sort(omega)
    vr = vr[:, omega_sort_index]
    omega = omega_sort/2/np.pi

    norm = np.sqrt(np.diag(vr.T@Mii@vr))
    phi = vr / norm

    return phi, omega 