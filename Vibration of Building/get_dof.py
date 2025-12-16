# Import packages
import numpy as np

def generate_dofs_and_find_indices(Nodes):
    """
    Generate a list of DOFs for all nodes and a mapping of DOF to its index.

    Parameters:
    - Nodes: A NumPy array of nodes where each row is [NodeID, X, Y, Z]

    Returns:
    - DOFs: A list of all DOFs in the format NodeID.DOF
    - dof_to_index: A dictionary mapping each DOF to its index in the DOFs list
    """
    nNode = Nodes.shape[0]
    DOFs = np.zeros(nNode * 6)
    
    for i in range(nNode):
        nodeID = Nodes[i, 0]
        for j in range(1, 7):  # 6 DOFs per node
            DOFs[i*6 + j-1] = nodeID + j/100  # nodeId.DOF
            
    dof_to_index = {dof: idx for idx, dof in enumerate(DOFs)}
    
    return DOFs, dof_to_index

def find_dof_index(DOFs, dof_to_index, dof):
    """
    Find the index of a specific DOF in the DOFs list.

    Parameters:
    - DOFs: The list of DOFs
    - dof_to_index: A dictionary mapping each DOF to its index
    - dof: The specific DOF to find the index for

    Returns:
    - The index of the specified DOF, if it exists
    """
    if dof in dof_to_index:
        return dof_to_index[dof]
    else:
        return None
    
def remove_dof(DOF, seldof):
    """
    Remove specified degrees of freedom representing Dirichlet boundary conditions set to zero.

    Parameters:
    - DOF: NumPy array of degrees of freedom.
    - seldof: NumPy array of selected degrees of freedom to be removed.

    Returns:
    - Modified array of degrees of freedom.
    """
    DOF = np.array(DOF).flatten()
    seldof = np.array(seldof).flatten()

    if np.any(seldof == 0.00):
        raise ValueError("The wild card 0.00 is not allowed")

    indj = np.ones(DOF.shape, dtype=bool)
    
    for dof in seldof:
        if int(dof) == 0:  # Wild cards 0.0X
            indjdof = np.abs(DOF % 1 - dof % 1) < 0.0001
        elif dof % 1 == 0:  # Wild cards X.00
            indjdof = np.abs(np.floor(DOF) - np.floor(dof)) < 0.0001
        else:  # Standard case
            indjdof = np.abs(DOF - dof) < 0.0001
        
        if not np.any(indjdof):
            raise ValueError(f"The degree of freedom {dof:.2f} does not exist")
        
        indj &= ~indjdof
    
    dof_to_index = {dof: idx for idx, dof in enumerate(DOF[indj])}

    return DOF[indj], dof_to_index