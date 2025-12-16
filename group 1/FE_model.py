"""
FE model simplified multi-storey building.

Features:
Two floor groups with distinct sections, material properties and kinematic support conditions. Columns are uniform over the height of the structure. 

"""

# Import packages
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# ---------------------------------------------- Dimension setting ---------------------------------------------- #
# Main structural dimensions
n_floor=5             # number of floors
H_col=3.5             # column height: distance between floors [m]
B_floor=6             # width of floors [m] -3 to 3
n_node_col=7          # number of (structural) nodes per column
n_node_floor=5        # number of (structural) nodes per floor element (excluding column nodes); odd number to allow excitation at floor centre

# Section dimensions:
# - section 1: left and right columns
b_col = 0.3           # width of column [m]
h_col = 0.3           # height of column [m]

# - section 2: floor group I (bottom two floors)
b_floor1 = 0.3        # width of beam [m]
h_floor1 = 0.5        # height of beam [m]
k_bot = 1             # kinematic constraint factor [-]

# - section 3: floor group II (top three floors)
b_floor2 = 0.25       # width of beam [m]
h_floor2 = 0.35       # height of beam [m]
k_top = 30            # kinematic constraint factor [-]

# Material properties:
# - material 1: M40 concrete (left and right columns) 
E_col = 40e9          # Young's modulus [N/m2]
rho_col = 2500        # concrete density [kg/m3]
nu_col = 0.25         # Poisson's ratio [-]

# - material 2: M35 concrete (floor group I)
E_floor1 = 35e9       # Young's modulus [N/m2]
rho_floor1 = 2500     # concrete density [kg/m3]
nu_floor1 = 0.25      # Poisson's ratio [-]
mass_floor1 = 5000   # mass per unit length [kg/m]

# - material 3: M30 concrete (floor group II)
E_floor2 = 30e9       # Young's modulus [N/m2]
rho_floor2 = 2200     # concrete density [kg/m3]
nu_floor2 = 0.25      # Poisson's ratio [-]
mass_floor2 = 5000   # mass per unit length [kg/m]

# floor groups
floor1 = [0, 1]       # floor numbers in floor bottoms
floor2 = [2, 3, 4]    # floor numbers in floor tops

# ---------------------------------------------- Properties setting ---------------------------------------------- #
# Property left and right columns:
dict_prop_col = ({'E': E_col, 'rho': rho_col, 'nu': nu_col,
                'b': b_col, 'h': h_col, 'node': n_node_col, 'H': H_col,
                'A': b_col * h_col, 'I': b_col * h_col ** 3 / 12})
# Property of floor group I (bottom two floors):
dict_prop_flr_bot = ({'E': E_floor1, 'rho': rho_floor1, 'nu': nu_floor1,
                    'b': b_floor1, 'h': h_floor1, 'node': n_node_floor, 'B': B_floor,
                    'A': b_floor1 * h_floor1, 'I': b_floor1 * h_floor1 ** 3 / 12, 'connect':k_bot})
# Property of floor group II (top three floors):
dict_prop_flr_top = ({'E': E_floor2, 'rho': rho_floor2, 'nu': nu_floor2,
                        'b': b_floor2, 'h': h_floor2, 'node': n_node_floor, 'B': B_floor,
                        'A': b_floor2 * h_floor2, 'I': b_floor2 * h_floor2 ** 3 / 12, 'connect':k_top})

# ---------------------------------------------- Node setting ---------------------------------------------- #
Node_xyz = np.zeros((0, 3))

# Left column nodes
for ifln in range(n_floor):
    y0 = dict_prop_col['H'] * ifln
    ye = y0 + dict_prop_col['H']

    if ifln == 0:
        Node_xyz = np.vstack((Node_xyz, [-B_floor / 2, 0, 0]))

    x_values = np.full(dict_prop_col['node']-1, -B_floor / 2)
    y_values = np.linspace(y0+(ye-y0)/(dict_prop_col['node']-1), ye, dict_prop_col['node'] - 1)
    z_values = np.zeros(dict_prop_col['node']-1)
    column_nodes = np.column_stack((x_values, y_values, z_values))

    Node_xyz = np.vstack((Node_xyz, column_nodes))

# id of column left nodes
n_id_coll = [i for i in range(n_floor * (dict_prop_col['node']-1) + 1)]
# id of column left nodes connected to floor
n_id_collf = [(n_node_col-1)*(i+1) for i in range(n_floor)]

# Right column nodes
for ifln in range(n_floor):
    y0 = dict_prop_col['H'] * ifln
    ye = y0 + dict_prop_col['H']

    if ifln == 0:
        Node_xyz = np.vstack((Node_xyz, [B_floor / 2, 0, 0]))

    x_values = np.full(dict_prop_col['node']-1, B_floor / 2)
    y_values = np.linspace(y0+(ye-y0)/(dict_prop_col['node']-1), ye, dict_prop_col['node'] - 1)
    z_values = np.zeros(dict_prop_col['node']-1)
    column_nodes = np.column_stack((x_values, y_values, z_values))

    Node_xyz = np.vstack((Node_xyz, column_nodes))

# id of column right nodes
n_id_colr = [i+n_id_coll[-1]+1 for i in range(n_floor * (dict_prop_col['node']-1) + 1)]
# id of column right nodes connected to floor
n_id_colrf = [(n_node_col-1)*(i+1)+n_id_coll[-1]+1 for i in range(n_floor)]


dict_n_id_floor = {}
# Floor nodes
for ifln in range(n_floor):
    y0 = dict_prop_col['H'] * ifln
    ye = y0 + dict_prop_col['H']
    x0 = -B_floor / 2
    xe = B_floor / 2
    dx = (xe - x0) / (n_node_floor + 1)

    Node_xyz = np.vstack((Node_xyz, [-B_floor / 2, ye, 0]))

    floor_nodes = np.column_stack(
        (np.arange(x0 + dx, xe, dx), np.full(n_node_floor, ye), np.zeros(n_node_floor)))

    Node_xyz = np.vstack((Node_xyz, floor_nodes))
    Node_xyz = np.vstack((Node_xyz, [B_floor / 2, ye, 0]))
    dict_n_id_floor[ifln] = [n_id_colr[-1]+1+(n_node_floor+2)*ifln+i for i in range(n_node_floor+2)]

# id of floor nodes
n_id_fl0, n_id_fl1, n_id_fl2, n_id_fl3, n_id_fl4 = [dict_n_id_floor[key] for key in dict_n_id_floor.keys()]
# id of floor nodes connected to column left and right
n_id_fll = [n_id_fl0[0], n_id_fl1[0], n_id_fl2[0], n_id_fl3[0], n_id_fl4[0]]
n_id_flr = [n_id_fl0[-1], n_id_fl1[-1], n_id_fl2[-1], n_id_fl3[-1], n_id_fl4[-1]]

# id of floor nodes
n_id_fl = n_id_fl0 + n_id_fl1 + n_id_fl2 + n_id_fl3 + n_id_fl4

# id of origin node
n_id_origin = Node_xyz.shape[0]

# Add origin node
Node_xyz = np.vstack((Node_xyz, [0, 0, 0]))

# Add node numbers
nNode = Node_xyz.shape[0]
NodID = np.arange(1, nNode + 1)[:, np.newaxis]  # Node numbers
Nodes = np.hstack((NodID, Node_xyz)) # Nodes = [NodeNumber X Y Z]

## Plot nodes:
#plt.figure(figsize=(4,12))
#plt.plot(Nodes[:, 1], Nodes[:, 2], 'bo', markersize=5)  
#for i in range(nNode):
#    plt.text(Nodes[i, 1], Nodes[i, 2], str(int(Nodes[i, 0])), fontsize=8, color='black', ha='center', va='bottom')
#plt.xlabel('x-coordinate')
#plt.ylabel('y-coordinate')
#plt.title('Nodes')
#plt.grid(True)
#plt.show()#

# ---------------------------------------------- Elements setting ---------------------------------------------- #
# Elements
n_elem = 0

# left column elements
e_coll = np.zeros((len(n_id_coll)-1, 8))
for i in range(len(n_id_coll)-1):
    rho = dict_prop_col['rho']   # [kg/m3]
    e_coll[i] = ([n_id_coll[i], n_id_coll[i+1],
                rho, dict_prop_col['A'], dict_prop_col['I'],  dict_prop_col['E']*dict_prop_col['A'],
                dict_prop_col['E']*dict_prop_col['I'], dict_prop_col['H']])
    # n_elem += 1

# right column elements
e_colr = np.zeros((len(n_id_colr)-1, 8))
for i in range(len(n_id_colr)-1):
    rho = dict_prop_col['rho']   # [kg/m]
    e_colr[i] = ([n_id_colr[i], n_id_colr[i+1],
                rho, dict_prop_col['A'], dict_prop_col['I'], dict_prop_col['E']*dict_prop_col['A'],
                dict_prop_col['E']*dict_prop_col['I'], dict_prop_col['H']])
    # n_elem += 1


# floor elements
# group 1
e_fl1 = np.zeros((len(dict_n_id_floor[0])+len(dict_n_id_floor[1])-2, 8))
for j in range(len(floor1)):
    for i in range(len(dict_n_id_floor[0])-1):
        rho = dict_prop_flr_bot['rho'] + mass_floor1/dict_prop_flr_bot['A']   # [kg/m3]
        e_fl1[i+j*(len(dict_n_id_floor[j])-1)] = ([dict_n_id_floor[j][i], dict_n_id_floor[j][i+1],
                    rho, dict_prop_flr_bot['A'], dict_prop_flr_bot['I'], dict_prop_flr_bot['E']*dict_prop_flr_bot['A'],
                    dict_prop_flr_bot['E']*dict_prop_flr_bot['I'], dict_prop_flr_bot['B']])
        # n_elem += 1


# group 2
e_fl2 = np.zeros((len(dict_n_id_floor[2])+len(dict_n_id_floor[3])+len(dict_n_id_floor[4])-3, 8))
for j in range(len(floor2)):
    for i in range(len(dict_n_id_floor[2])-1):
        rho = dict_prop_flr_top['rho'] + mass_floor2/dict_prop_flr_top['A']   # [kg/m3]
        e_fl2[i+j*(len(dict_n_id_floor[j])-1)] = ([dict_n_id_floor[j+2][i], dict_n_id_floor[j+2][i+1],
                    rho, dict_prop_flr_top['A'], dict_prop_flr_top['I'], dict_prop_flr_top['E']*dict_prop_flr_top['A'],
                    dict_prop_flr_top['E']*dict_prop_flr_top['I'], dict_prop_flr_top['B']])
        # n_elem += 1


# global element matrix
e_all = np.vstack((e_coll, e_colr, e_fl1, e_fl2))
n_elem = e_all.shape[0]

# Plot structure (Structure with nodes and element indices)
plt.figure(figsize=(8,16))
overlapping_nodes = {}
for i in range(nNode):
    coord = (Nodes[i, 1], Nodes[i, 2])
    if coord not in overlapping_nodes:
        overlapping_nodes[coord] = [str(int(Nodes[i, 0]))]
    else:
        overlapping_nodes[coord].append(str(int(Nodes[i, 0])))

plt.plot(Nodes[:, 1], Nodes[:, 2], 'ko', markersize=3)
for coord, indices in overlapping_nodes.items():
    text_annotation = ';'.join(indices)
    plt.text(coord[0], coord[1], text_annotation, fontsize=8, color='k', ha='right', va='bottom')    

for iElem in np.arange(0, n_elem):
    NodeLeft = int(e_all[iElem][0])
    NodeRight = int(e_all[iElem][1])
    plt.plot([Nodes[NodeLeft][1], Nodes[NodeRight][1]], [Nodes[NodeLeft][2], Nodes[NodeRight][2]], 'b',linewidth=0.5)
    # Plot element indices
    x_mid = (Nodes[NodeLeft][1] + Nodes[NodeRight][1]) / 2
    y_mid = (Nodes[NodeLeft][2] + Nodes[NodeRight][2]) / 2
    plt.text(x_mid, y_mid, str(iElem+1), fontsize=8, color='b', ha='center', va='center')

plt.axis('equal')
plt.xlabel('x-coordinate')
plt.ylabel('y-Coordinate')
plt.title('Structure with nodes and element Indices')
plt.grid()
plt.show()

# ---------------------------------------- Boundary conditions and DOFs ----------------------------------------- #
# 2D analysis: 3 DOFs per node (2D analysis: UX,UY,ROTZ)
# DOFs: A list of all DOFs in the format NodeID.DOF
#           DOFs -> [1.01 1.02 1.03 2.01 ...].'
#           1.01 represents the 1st DOF (UX) of node 1
#           1.02 represents the 2nd DOF (UY) of node 1
#           1.03 represents the 6st DOF (ROTZ) of node 1
#           2.01 represents the 1st DOF (UX) of node 2

DOF = np.zeros(nNode * 3) 

for i in range(nNode):
    nodeID = Nodes[i, 0]
    for j in range(1, 4):                # 3 DOFs per node (2D analysis: UX,UY,ROTZ)
        DOF[i*3 + j-1] = nodeID + j/100  # nodeId.DOF

DOF = np.array(DOF).flatten()
removedof = np.array([1, 32, 98])        # clamp / remove all dof at bottom: node 1, 32 and reference node
removedof = np.array(removedof).flatten()
indj = np.ones(DOF.shape, dtype=bool)

for dof in removedof:
    if int(dof) == 0:    # Wild cards 0.0X
        indjdof = np.abs(DOF % 1 - dof % 1) < 0.0001
    elif dof % 1 == 0:   # Wild cards X.00
        indjdof = np.abs(np.floor(DOF) - np.floor(dof)) < 0.0001
    else:                # Standard case
        indjdof = np.abs(DOF - dof) < 0.0001
    
    if not np.any(indjdof):
        raise ValueError(f"The degree of freedom {dof:.2f} does not exist")
    
    indj &= ~indjdof

dof_to_index = {dof: idx for idx, dof in enumerate(DOF[indj])}
ndofs = len(dof_to_index) # number of dofs
DOFs = DOF[indj]          # DOFs: A list of all DOFs in the format NodeID.DOF

# To find the index of a specific DOF, e.g., 2.02:
dof = 2.01
if dof in dof_to_index:
    DOF_index = dof_to_index[dof]

# ---------------------------------------- DOF of sensor locations ----------------------------------------- #
# get the corresponding DOF of sensor locations
sensor_locations = pd.read_csv('sensor_locations.csv')
S_d_array = sensor_locations.iloc[:, 1:].to_numpy()

n_d = S_d_array.shape[0]
sensor_dofs = []

for ind in range(n_d):
    measured_dofs = np.where(S_d_array[ind, :] == 1)[0]  # Indices of measured DOFs for current sensor
    for dof_index in measured_dofs:
        sensor_dofs.append(DOFs[dof_index])

sensor_dofs = np.array(sensor_dofs)

# Find DOF directions for plotting
# -> 0.01 = Ux (horizontal measurement)
# -> 0.02 = Uy (vertical measurement)
direction = np.round((S_d_array.dot(DOFs) % 1) * 100).astype(int) 

# ---------------------------------------------- Mass and Stiffness matrices --------------------------------------------------------------- #
n_dofs = 3 * (Node_xyz.shape[0]-1)    # number of degrees of freedom, 3 per node: x and y displacement and z rotation
                                      # minus 1 because removing the origin node
K = np.zeros((n_dofs*n_dofs))         # global stiffness matrix
M = np.zeros((n_dofs*n_dofs))         # global mass matrix

def BeamMatricesJacket(rho, A, I, EA, EI, NodeCoord):
    """
    # Inputs:
    # rho       - density [kg/m3]
    # A         - cross-sectional area [m2]
    # m         - mass per unit length [kg/m]
    # I         - second moment of area [m4]
    # EA        - axial stiffness [N]
    # EI        - bending stiffness [N.m2]
    # NodeCoord - ([xl, yl], [xr, yr])      
    #           - left (l) and right (r) node coordinates
    """

    # 1 - calculate length of beam (L) and orientation alpha
    xl = NodeCoord[0][0]    # x-coordinate of left node
    yl = NodeCoord[0][1]    # y-coordinate of left node
    xr = NodeCoord[1][0]    # x-coordinate of right node
    yr = NodeCoord[1][1]    # y-coordinate of rigth node
    L = np.sqrt((xr - xl)**2 + (yr - yl)**2)    # length
    alpha = math.atan2(yr - yl, xr - xl)        # angle

    # 2 - calculate transformation matrix T
    C = np.cos(alpha)
    S = np.sin(alpha)
    T = np.array([[C, -S, 0], [S, C, 0], [0, 0, 1]])
    T = np.asarray(np.bmat([[T, np.zeros((3,3))], [np.zeros((3, 3)), T]]))
    

    # 3 - calculate local stiffness and matrices
    L2 = L*L
    L3 = L*L2
    # Local stiffness matrix
    K = np.array([[EA/L, 0, 0, -EA/L, 0, 0], 
                    [0, 12*EI/L3, 6*EI/L2, 0, -12*EI/L3, 6*EI/L2], 
                    [0, 6*EI/L2, 4*EI/L, 0, -6*EI/L2, 2*EI/L], 
                    [-EA/L, 0, 0, EA/L, 0, 0], 
                    [0, -12*EI/L3, -6*EI/L2, 0, 12*EI/L3, -6*EI/L2], 
                    [0, 6*EI/L2, 2*EI/L, 0, -6*EI/L2, 4*EI/L]])
    
    # Not used anymore
    M = (rho*A*L2/420) *np.array([[140, 0, 0, 70, 0, 0], 
                            [0, 156, 22*L, 0, 54, -13*L], 
                            [0, 22*L, 4*L2, 0, 13*L, -3*L2], 
                            [70, 0, 0, 140, 0, 0], 
                            [0, 54, 13*L, 0, 156, -22*L], 
                            [0, -13*L, -3*L2, 0, -22*L, 4*L2]])
    
    # Local mass matrix
    M_mat = np.zeros((6,6))
    M_mat[0,0] = rho*A*L/3
    M_mat[1,1] = rho*(13*A*L**2 + 42*I)/(35*L)
    M_mat[2,2] = rho*L*(A*L**2 + 14*I)/105
    M_mat[3,3] = M_mat[0,0]
    M_mat[4,4] = M_mat[1,1]
    M_mat[5,5] = M_mat[2,2]
    M_mat[0,3] = rho*A*L/6
    M_mat[3,0] = M_mat[0,3]
    M_mat[2,1] = rho*(11*A*L**2 + 21*I)/(210)
    M_mat[1,2] = M_mat[2,1]
    M_mat[5,4] = -M_mat[2,1]
    M_mat[4,5] = M_mat[5,4]
    M_mat[4,1] = rho*(9*A*L**2 - 84*I)/(70*L)
    M_mat[1,4] = M_mat[4,1]
    M_mat[4,2] = rho*(13*A*L**2 - 42*I)/(420)
    M_mat[2,4] = M_mat[4,2]
    M_mat[5,1] = -M_mat[4,2]
    M_mat[1,5] = M_mat[5,1]
    M_mat[5,2] = -rho*L*(3*A*L**2 + 14*I)/(420)
    M_mat[2,5] = M_mat[5,2]


    # 4 - rotate the matrices
    K = np.matmul(T, np.matmul(K, np.transpose(T)))
    M = np.matmul(T, np.matmul(M_mat, np.transpose(T)))
    return M, K

for iElem in np.arange(0, n_elem):
    # Get the nodes of the elements
    NodeLeft = int(e_all[iElem][0])
    NodeRight = int(e_all[iElem][1])
    
    # Get the coordinates of the nodes of the element
    coord = [Node_xyz[NodeLeft], Node_xyz[NodeRight]]

    # Get the degrees of freedom that correspond to each node
    Dofs_Left = 3*(NodeLeft) + np.arange(0, 3)
    Dofs_Right = 3*(NodeRight) + np.arange(0, 3)

    # Get the properties of the element
    rho = e_all[iElem][2]
    A = e_all[iElem][3]
    I = e_all[iElem][4]
    EA = e_all[iElem][5]
    EI = e_all[iElem][6]
    
    # Calculate the matrices of the element
    Me, Ke = BeamMatricesJacket(rho, A, I, EA, EI, coord)

    # Assemble the matrices at the correct place
    nodes = np.append(Dofs_Left, Dofs_Right)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            ij = nodes[j] + nodes[i]*n_dofs
            M[ij] = M[ij] + Me[i, j]
            K[ij] = K[ij] + Ke[i, j]

M_before = M.copy()
K_before = K.copy()
# Reshape the global matrix from a 1-dimensional array to a 2-dimensional array
M = M.reshape((n_dofs, n_dofs))
K = K.reshape((n_dofs, n_dofs))


# Node id 0 and 31 is fixed: node fixed to the groud
DOF_fixed = np.array([0, 1, 2, 31*3, 31*3+1, 31*3+2])
DOF_free = np.arange(0, n_dofs)
DOF_free = np.delete(DOF_free, DOF_fixed)

# print(DOF_free)
fx = DOF_free[:, np.newaxis]
fy = DOF_free[np.newaxis, :]
bx = DOF_fixed[:, np.newaxis]
by = DOF_fixed[np.newaxis, :]

# Mass
Mii = M[fx, fy]    # For DOF_free
Mib = M[fx, by]
Mbi = M[bx, fy]
Mbb = M[bx, by]

# Stiffness
Kii = K[fx, fy]    # For DOF_free
Kib = K[fx, by]
Kbi = K[bx, fy]
Kbb = K[bx, by]

df_Mii = pd.DataFrame(Mii)
df_Kii = pd.DataFrame(Kii)

# df_Kii.to_csv('Kii.csv', index=False, header=False)
# df_Mii.to_csv('Mii.csv', index=False, header=False)

n_DOF_free = len(DOF_free)

#plt.figure()
#plt.spy(Mii)
#plt.title("Mass matrix")
#plt.figure()
#plt.spy(Kii)
#plt.title("Stiffness matrix")
#plt.show()

# ---------------------------------------------- Kinematic constraints ---------------------------------------------- #
n_cstr = 0
constr = np.empty((0, 5))
# Kinematic constraints for the column on the left side
# make it able to set constraints per floor group (bottom 2 floors, top 3 floors)
# [Constant CoefS SlaveDOF CoefM1 MasterDOF1 CoefM2 MasterDOF2 ...]
# [0,  1,                    Nfcl(i)+0.01,    -1,   Nccl(i)+0.01 ]
for i in range(len(n_id_collf)):
    # get node id and DOF of the column
    node = n_id_collf[i]
    DOF_col = 3*node - 3 + np.arange(0, 3)    # because node 0 removed in Mii and Kii, -3
    # get node id and DOF of the correspoinding floor
    node_floor = n_id_fll[i]
    DOF_floor = 3*node_floor - 6 + np.arange(0, 3)   # because 2 nodes removed in Mii and Kii, -6

    n_cstr += 1
    if i == 0 or i == 1:
        con_child_x = np.array([0, 1, DOF_floor[0], -1, DOF_col[0]])
        con_child_y = np.array([0, 1, DOF_floor[1], -1, DOF_col[1]])
        con_child_z = np.array([0, dict_prop_flr_bot['connect'], DOF_floor[2], -1, DOF_col[2]])
        con_child = np.vstack((con_child_x, con_child_y, con_child_z))
        try:
            constr = np.vstack((constr, con_child))
        except NameError:
            constr = np.array(con_child)
    elif i == 2 or i == 3 or i == 4:
        con_child_x = np.array([0, 1, DOF_floor[0], -1, DOF_col[0]])
        con_child_y = np.array([0, 1, DOF_floor[1], -1, DOF_col[1]])
        con_child_z = np.array([0, dict_prop_flr_top['connect'], DOF_floor[2], -1, DOF_col[2]])
        con_child = np.vstack((con_child_x, con_child_y, con_child_z))

        constr = np.vstack((constr, con_child))

# Kinematic constraints for the column on the right side
for i in range(len(n_id_colrf)):
    # get node id and DOF of the column
    node = n_id_colrf[i]
    DOF_col = 3*node - 6 + np.arange(0, 3)    # because 2 nodes removed in Mii and Kii, -6
    # get node id and DOF of the correspoinding floor
    node_floor = n_id_flr[i]
    DOF_floor = 3*node_floor - 6 + np.arange(0, 3)    # because 2 nodes removed in Mii and Kii, -6

    n_cstr += 1
    if i == 0 or i == 1:
        con_child_x = np.array([0, 1, DOF_floor[0], -1, DOF_col[0]])
        con_child_y = np.array([0, 1, DOF_floor[1], -1, DOF_col[1]])
        con_child_z = np.array([0, dict_prop_flr_bot['connect'], DOF_floor[2], -1, DOF_col[2]])
        con_child = np.vstack((con_child_x, con_child_y, con_child_z))
        constr = np.vstack((constr, con_child))
    elif i == 2 or i == 3 or i == 4:
        con_child_x = np.array([0, 1, DOF_floor[0], -1, DOF_col[0]])
        con_child_y = np.array([0, 1, DOF_floor[1], -1, DOF_col[1]])
        con_child_z = np.array([0, dict_prop_flr_top['connect'], DOF_floor[2], -1, DOF_col[2]])
        con_child = np.vstack((con_child_x, con_child_y, con_child_z))

        constr = np.vstack((constr, con_child))

# Apply constraints to the matrices
Mii_con = Mii.copy()
Kii_con = Kii.copy()

n_cstr = constr.shape[0]

Constant = constr[:, 0] / constr[:, 1]
slaveDOF = constr[:, 2].astype(int)
masterDOF = constr[:, 4].astype(int)
coeff = - constr[:, 3] / constr[:, 1]

for i in range(constr.shape[0]):
    Mii_con[masterDOF[i], :] = Mii_con[masterDOF[i], :] + coeff[i] * Mii_con[slaveDOF[i], :]
    Mii_con[:, masterDOF[i]] = Mii_con[:, masterDOF[i]] + coeff[i] * Mii_con[:, slaveDOF[i]]
    Kii_con[masterDOF[i], :] = Kii_con[masterDOF[i], :] + coeff[i] * Kii_con[slaveDOF[i], :]
    Kii_con[:, masterDOF[i]] = Kii_con[:, masterDOF[i]] + coeff[i] * Kii_con[:, slaveDOF[i]]

    # Remove slave DOF
    Mii_con[slaveDOF[i], :] = 0
    Mii_con[:, slaveDOF[i]] = 0
    Kii_con[slaveDOF[i], :] = 0
    Kii_con[:, slaveDOF[i]] = 0
    Kii_con[slaveDOF[i], slaveDOF[i]] = 1
    Kii_con[slaveDOF[i], masterDOF[i]] = -coeff[i]

# -------------------------------------------------------Calculate eigenfrequencies and eigenmodes------------------------------------------------------- #
mat = np.dot(np.linalg.pinv(Mii_con), Kii_con)
w2, Phi_calculated = np.linalg.eig(mat)

# Calculate eigenfrequencies and then sort them in an ascending order
Omega_calculated = np.sqrt(np.abs(w2))
index_omega_calculated = Omega_calculated.argsort()[::1]
Omega_calculated_sorted = Omega_calculated[index_omega_calculated]
Omega_calculated_sorted = Omega_calculated_sorted[n_cstr:]

Phi_calculated_sorted = Phi_calculated[:, index_omega_calculated]
Phi_calculated_sorted = Phi_calculated_sorted[:, n_cstr:]
Phi_calculated_sorted = Phi_calculated_sorted.real.astype(float)

# Mass-normalise the eigenmodes
phi_transpose = np.transpose(Phi_calculated_sorted)
n = np.sqrt(np.diag(np.dot(np.dot(phi_transpose, Mii_con), Phi_calculated_sorted)))
Mass_Normalized_phi = (Phi_calculated_sorted / np.tile(n, (Phi_calculated_sorted.shape[0], 1))).astype(float)

Omega = Omega_calculated_sorted 
Phi = Mass_Normalized_phi

# Display results
print("Eigenfrequencies [Hz]:\n")
print(Omega_calculated_sorted[:10]/(2*np.pi))

# -------------------------------------------------------Plot modes------------------------------------------------------- #
nomodes = 10 #number of modes to plot

modes = np.arange(1, nomodes + 1)
Phi_plot = 3 * Phi
for i in modes:
    
    # set initial matrices for animation
    x_left = np.zeros((nomodes, e_all.shape[0]))
    y_left = np.zeros((nomodes, e_all.shape[0]))
    x_right = np.zeros((nomodes, e_all.shape[0]))
    y_right = np.zeros((nomodes, e_all.shape[0]))


    for mode in range(nomodes):
        # Select the mode to plot
        Phi_part = Phi_plot[:,mode]

        # DOF in x direction
        x_part = Phi_part[::3]

        # add fixed nodes (node 0 and node 31)
        x_part1 = x_part[:30].reshape(-1, 1)
        x_part2 = x_part[30:].reshape(-1, 1)
        fixed_node = np.zeros((1,1))
        x_all = np.vstack((fixed_node, x_part1, fixed_node, x_part2))

        # DOF in y direction
        y_part = Phi_part[1::3]

        # add fixed nodes (node 0 and node 31)
        y_part1 = y_part[:30].reshape(-1, 1)
        y_part2 = y_part[30:].reshape(-1, 1)
        fixed_node = np.zeros((1,1))
        y_all = np.vstack((fixed_node, y_part1, fixed_node, y_part2))

        # connection nodes of columns
        parent_left = [6*i for i in range(1,6)]
        parent_right = [6*i+31 for i in range(1,6)]

        # replace values of floor nodes with the values of the parent node
        for col in parent_left:
            child = int(7/6*col + 55)
            x_all[child] = x_all[int(col)]
            y_all[child] = y_all[int(col)]

        for col in parent_right:
            child = int(7/6*col + 149/6)
            x_all[child] = x_all[int(col)]
            y_all[child] = y_all[int(col)]
    

        for iElem in np.arange(0, e_all.shape[0]):
            NodeLeft = int(e_all[iElem][0])
            NodeRight = int(e_all[iElem][1])
            # print(NodeLeft, NodeRight)
            # print(Node_xyz[NodeLeft][0], Node_xyz[NodeRight][0])
            # print(NodeRight)
            m = e_all[iElem][2]
            EA = e_all[iElem][3]
            EI = e_all[iElem][4]
            # (Node_xyz[NodeRight][0]-Node_xyz[NodeLeft][0])
            x_left[mode, iElem] = Node_xyz[NodeLeft][0] + 200*x_all[NodeLeft][0] * np.exp(1j * 0 * np.pi/60).real
            y_left[mode, iElem] = Node_xyz[NodeLeft][1] + 200*y_all[NodeLeft][0] * np.exp(1j * 0 * np.pi/60).real
            x_right[mode, iElem] = Node_xyz[NodeRight][0] + 200*x_all[NodeRight][0] * np.exp(1j * 0 * np.pi/60).real
            y_right[mode, iElem] = Node_xyz[NodeRight][1] + 200*y_all[NodeRight][0] * np.exp(1j * 0 * np.pi/60).real

    
    plt.figure(figsize=(5,5))
    for iElem in np.arange(0, e_all.shape[0]):
        plt.plot([x_left[i-1, iElem], 
                    x_right[i-1, iElem]], 
                [y_left[i-1, iElem], 
                    y_right[i-1, iElem]], 
                c='b', alpha=0.5)
    
    for iElem in np.arange(0, e_all.shape[0]):
        NodeLeft = int(e_all[iElem][0])
        NodeRight = int(e_all[iElem][1])
        m = e_all[iElem][2]
        EA = e_all[iElem][3]
        EI = e_all[iElem][4]
        if iElem < (len(e_coll) + len(e_colr)):
            plt.plot([Node_xyz[NodeLeft][0], Node_xyz[NodeRight][0]], [Node_xyz[NodeLeft][1], Node_xyz[NodeRight][1]], c='k', alpha=0.4)
        elif iElem < (len(e_coll) + len(e_colr) + len(e_fl1)):
            plt.plot([Node_xyz[NodeLeft][0], Node_xyz[NodeRight][0]], [Node_xyz[NodeLeft][1], Node_xyz[NodeRight][1]], c='k', alpha=0.4)
        else:
            plt.plot([Node_xyz[NodeLeft][0], Node_xyz[NodeRight][0]], [Node_xyz[NodeLeft][1], Node_xyz[NodeRight][1]], c='k', alpha=0.4)
    plt.axis('equal')
    
    plt.xlim(-10, 10)
    plt.ylim(-0.5,20)
    omega = Omega[i-1]/(2*np.pi)
    plt.title(f'Calculated mode {str(i)} - {omega:.4f} Hz')
    plt.grid()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show()   