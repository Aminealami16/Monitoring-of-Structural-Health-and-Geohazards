# Import packages
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------- Define general properties -------------------------------------------- #
g = 9.81              # [m/s^2]

# Define properties of the beam 
b = 0.050             # [m]
h = 0.025             # [m]
L = 1                 # [m]
rho = 7750            # [kg/m^3]
A = b*h               # [m^2]
E = 210e9             # [Pa]
I = 1/12*b*h**3       # [m^4]


def get_M_and_K_matrix():
    # ---------------------------------------------------- Create mesh --------------------------------------------------- #
    # Total number of elements
    n_elements = 20

    # Total number of nodes, nn = ne + 1
    n_nodes = n_elements + 1

    # Total number of degrees of freedom, 2 per node, 1 for displacement and 1 for rotation
    n_dofs = 2*n_nodes

    # Create a vector with the coordinates of the nodes
    x_cordinates_nodes = np.linspace(0, L, n_nodes)
    
    # Create empty lists for the elementall degrees of freedom, so we can specify which nodes are connected to each element
    elem_dofs = []
    # Create empty lists for the nodeall degrees of freedom, so we can specify which degrees of freedom each node has
    dof_node = []

    # Loop over the elements and append the the elementall degrees of freedom
    for ie in np.arange(0, n_elements):
        elem_dofs.append(np.arange(2*ie, 2*ie + 4))

    # Loop over nodes and append the specific nodeall degrees of freedom to each node   
    for idof in np.arange(0, n_dofs):
        dof_node.append(int(np.floor(idof/2)))
        
    
    # ----------------------------------------------------- Plot mesh ---------------------------------------------------- #
    # Plot the mesh, every dot represents a node and every line represents an element
    nodes = (x_cordinates_nodes, np.ones(n_nodes))
    plt.figure(figsize=(20, 3))
    plt.title('Mesh of Euler-Bernoulli beam')
    plt.yticks([])
    plt.xticks(x_cordinates_nodes)
    plt.xlabel('x coordinate [m]')
    plt.plot(nodes[0], nodes[1],"o-");
    
    # ---------------------------------------------- Create shape functions and integrate ---------------------------------------------- #
    N_k = []
    dN_k = []
    ddN_k = []
    h = L/n_elements
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, h,  h**2, h**3], [0, 1, 2*h, 3*h**2]])

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
        
    import scipy.integrate as scp
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
        
    # ---------------------------------- Use integrated shape functions to fill matrices --------------------------------- #
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

        # Assemble the matrices
        nodes = np.append(Dofs_Left, Dofs_Right)
        for i in np.arange(0, 4):
            for j in np.arange(0, 4):
                ij = nodes[j] + nodes[i]*n_dofs
                M[ij] = M[ij] + M_k[i, j]
                K[ij] = K[ij] + K_k[i, j]
                
    # Reshape the global matrix from a 1-D array to a 2-D array
    M = M.reshape((n_dofs, n_dofs))
    K = K.reshape((n_dofs, n_dofs))
        
    # Prescribe boundary conditions
    DofsP = np.arange(0, 2)          # prescribed DOFs, clamp the left side by setting the first two DOFs of the leftmost node to zero
    DofsF = np.arange(0, n_dofs)     # free DOFs, notice all DOFs except the ones at the clamped side are free
    DofsF = np.delete(DofsF, DofsP)  # remove the fixed DOFs from the free DOFs array

    # free & fixed array indices, these are obtained to reshape the matrices
    fx = DofsF[:, np.newaxis]
    fy = DofsF[np.newaxis, :]

    # Mass
    Mii = M[fx, fy]

    # Stiffness
    Kii = K[fx, fy]
    
    # --------- Calculate eigenfrequencies and eigenmodes ---------------------------------------------------------------- #
    mat = np.dot(np.linalg.inv(Mii), Kii)
    w2, PHI_calculated = np.linalg.eig(mat)


    # Calculate eigenfrequencies and then sort them in an ascending order
    Omega_calculated = np.sqrt(w2.real)
    index_omega_calculated = Omega_calculated.argsort()
    Omega_calculated_sorted = Omega_calculated[index_omega_calculated]
    print("Eigenfrequencies [Hz]:\n")
    print(Omega_calculated_sorted[:10]/(2*np.pi))

    PHI_calculated_sorted = PHI_calculated[:,index_omega_calculated]


    # ----------------------------------------------- Mass-normalize the eigenmodes -------------------------------------- #
    phi_transpose = np.transpose(PHI_calculated_sorted)
    n = np.sqrt(np.diag(np.dot(np.dot(phi_transpose, Mii), PHI_calculated_sorted)))

    # Normalize phi
    Mass_Normalized_PHI = PHI_calculated_sorted / np.tile(n, (PHI_calculated_sorted.shape[0], 1))

    Omega = Omega_calculated_sorted
    Phi = Mass_Normalized_PHI
    
    
    # ----------------------------------------------- Create damping matrix ---------------------------------------------- #
    n_dofs_free = n_dofs - 2
    
    return Mii, Kii, Phi, Omega, n_elements, n_nodes, n_dofs_free # Need to get rid of fixed boundary DoF


def Mode_animation_plotter(mode_1, mode_2, mode_3, n_nodes_free):
    # defining the number of frames of the animations
    frame = int(40)
    mode_1_anim = np.zeros((20, frame))
    mode_2_anim = np.zeros((20, frame))
    mode_3_anim = np.zeros((20, frame))

    # Creating a sample sinusoidal wave to display the eigenmodes
    for i in range(frame):
        mode_1_anim[:, i] = mode_1*np.exp(1j*i*np.pi/20).real
        mode_2_anim[:, i] = mode_2*np.exp(1j*i*np.pi/20).real
        mode_3_anim[:, i] = mode_3*np.exp(1j*i*np.pi/20).real
    
    from ipywidgets import interact, fixed, widgets

    def mode_plot(x, u, step, title):
        ax = plt.axes(xticks=[], yticks=[], ylim=(u.min()*1.5, u.max()*1.5))
        ax.plot(x, u[:, step])
        ax.set_title("Mode " + str(title))
        plt.show()

    # animation of the first mode   
    play = widgets.Play(min=0, max=frame-1, step=1, value=0, interval=100, disabled=False)
    slider = widgets.IntSlider(min=0, max=frame-1, step=1, value=0)
    widgets.jslink((play, 'value'), (slider, 'value'))
    interact(mode_plot,
            x=fixed(np.arange(1, n_nodes_free+1)),
            u=fixed(mode_1_anim),
            step = play,
            title = fixed(1))
    widgets.HBox([slider])

    # animation of the second mode
    play = widgets.Play(min=0, max=frame-1, step=1, value=0, interval=100, disabled=False)
    slider = widgets.IntSlider(min=0, max=frame-1, step=1, value=0)
    widgets.jslink((play, 'value'), (slider, 'value'))
    interact(mode_plot,
            x=fixed(np.arange(1, n_nodes_free+1)),
            u=fixed(mode_2_anim),
            step = play,
            title = fixed(2))
    widgets.HBox([slider])

    # animation of the third mode
    play = widgets.Play(min=0, max=frame-1, step=1, value=0, interval=100, disabled=False)
    slider = widgets.IntSlider(min=0, max=frame-1, step=1, value=0)
    widgets.jslink((play, 'value'), (slider, 'value'))
    interact(mode_plot,
            x=fixed(np.arange(1, n_nodes_free+1)),
            u=fixed(mode_3_anim),
            step = play,
            title = fixed(3))
    widgets.HBox([slider])

    
    
    