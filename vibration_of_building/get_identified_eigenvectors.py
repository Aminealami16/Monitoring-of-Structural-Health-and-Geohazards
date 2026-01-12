"""
Select eigenfrequencies and corresponding eigenvectors from singular value spectrum and accompanying left singular vector matrices and plot

"""

# Import packages
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from FE_model import DOFs, Nodes, e_all

# Identified eigenfrequencies from singular value spectrum
freq1 = 0.32
freq2 = 3.08
freq3 = 4.84
freq4 = 9.66
freq5 = 13.75

# Create freq_id array
freq_seg = np.load('vibration_of_building/freq_seg.npy')
freq_id = np.array([freq1, freq2, freq3, freq4, freq5])
indx = [np.argmax(freq_seg > freq_val) for freq_val in freq_id]

# In order to visualize the identified eigenvectors we need to know the coordinates and directions of the measured DOFs. 
# ---------------------------------------------------------------------------------------------------------------------# 
sensor_locations = pd.read_csv('vibration_of_building/sensor_locations.csv')
S_d = sensor_locations.iloc[:, :].to_numpy()
n_d = S_d.shape[0]                           # number of sensors

NodeNr = []
NodeCo = []
ind_d = []
    
for ind in range(n_d):
    ind_d1 = np.where(S_d[ind, :] == 1)[0]      # Indices of measured DOFs for current sensor
    ind_d.append(ind_d1)
    NodeNr.append(np.floor(DOFs[ind_d1])-1)     # Node numbers
    NodeCo.append(Nodes[int(NodeNr[-1]), :])   # Nodal coordinates
    
ind_d = np.ravel(np.concatenate(np.array(ind_d), axis=0))

# Find DOF directions for plotting
direction = np.round((S_d.dot(DOFs) % 1) * 100).astype(int) 

# Select, normalize and plot:
# --------------------------#
U_omega = np.load('vibration_of_building/U_omega.npy')
Phi_id = np.zeros((n_d, len(indx)))
for ind in range(len(indx)):
    Phi_id[:, ind] = U_omega[indx[ind], :, 0].real
    Phi_id[:, ind] = Phi_id[:, ind] / np.linalg.norm(Phi_id[:, ind])

    node_number_to_index = {int(node_number[0]): index for index, node_number in enumerate(NodeNr)}
    
    # Define the side nodes and middle nodes for each floor using the new indices
    floor_info = {
        1: {'sides': [node_number_to_index[6], node_number_to_index[37]], 'middle': [node_number_to_index[n] for n in range(63, 68)]},
        2: {'sides': [node_number_to_index[12], node_number_to_index[43]], 'middle': [node_number_to_index[n] for n in range(70, 75)]},
        3: {'sides': [node_number_to_index[18], node_number_to_index[49]], 'middle': [node_number_to_index[n] for n in range(77, 82)]},
        4: {'sides': [node_number_to_index[24], node_number_to_index[55]], 'middle': [node_number_to_index[n] for n in range(84, 89)]},
        5: {'sides': [node_number_to_index[30], node_number_to_index[61]], 'middle': [node_number_to_index[n] for n in range(91, 96)]},
    }
    
    # Calculate the mean horizontal acceleration for side nodes outside the loop
    mean_horizontal_displacements = []
    for floor, info in floor_info.items():
        side_indices = info['sides']
        mean_horizontal_acceleration = np.mean([Phi_id[side_index, ind] for side_index in side_indices], axis=0)
        mean_horizontal_displacements.append(mean_horizontal_acceleration)

    NodeCo1 = np.copy(NodeCo)  # Start with original node coordinates
    for i in range(n_d):
        if direction[i] == 1:    # Horizontal measurement
            NodeCo1[i] += [0, Phi_id[i, ind], 0, 0]
        elif direction[i] == 2:  # Vertical measurement
            NodeCo1[i] += [0, 0, Phi_id[i, ind], 0]

        # Uniformly apply the mean horizontal displacement for middle nodes of each floor
        for floor_num, (floor, info) in enumerate(floor_info.items(), start=1):
            if i in info['middle']:
                NodeCo1[i][1] += mean_horizontal_displacements[floor_num - 1]  # Apply the mean displacement

    # Plot
    plt.figure()
    plt.title(f'Identified mode {ind + 1}')
    plt.axis('equal')
    for iElem in np.arange(0, e_all.shape[0]):
        NodeLeft = int(e_all[iElem][0])
        NodeRight = int(e_all[iElem][1])
        plt.plot([Nodes[NodeLeft][1], Nodes[NodeRight][1]], [Nodes[NodeLeft][2], Nodes[NodeRight][2]], 'b',linewidth=0.5)
    plt.plot(NodeCo1[:, 1], NodeCo1[:, 2], 'x',color='r', label='Mode Shape')
    plt.legend()
    plt.xlim(-10, 10)
    plt.ylim(-0.5,20)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.show()

# Save identified eigendata for model updating step
np.save('vibration_of_building/identified_eigdata.npy', {'Phi_id': Phi_id, 'freq_id': freq_id, 'ind_d': ind_d})