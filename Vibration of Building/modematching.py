def modematching(Phi_id, freq_id, Phi_m, freq_m):
    
    """
    Input arguments
    -  Phi_id:  measured/identified modes - normalized
    -  freq_id: measured/identified frequencies [Hz]
    -  Phi:     modelled modes  - normalized
    -  freq:    modelled frequencies [Hz]
    
    Output arguements: 
    -  modepairs: list relating pairs of corresponding modes [identified modelled]
    -  Phi_ids:  sorted identified modes 
    -  freq_ids: sorted identified frequencies
    -  Phi_s:    sorted modelled modes
    -  freq_s:   sorted modelled frequencies
    
    """
    
    # Import packages
    import numpy as np
    import pandas as pd
    import math
    import matplotlib.pyplot as plt

    # Compute MAC of identified and modelled modes
    MAC = np.zeros((Phi_id.shape[1] + 5, Phi_id.shape[1]))   # Initialize MAC matrix
    for i in range(Phi_id.shape[1] + 5):                     # Index over computed modes (larger)
        for j in range(Phi_id.shape[1]):                     # Index over identified modes
            numerator = np.abs(np.conj(Phi_m[:, i]) @ Phi_id[:, j])**2
            denominator = (np.conj(Phi_m[:, i]) @ Phi_m[:, i]) * (np.conj(Phi_id[:, j]) @ Phi_id[:, j])
            MAC[i, j] = numerator / denominator

    # Set the Threshold for allowing mode pairing
    ThMAC = 0.75

    # Pairing
    modepairs = np.array([], dtype=int).reshape(0, 2)           # preallocate modepairs
    for i in range(MAC.shape[1]):                               # for the number of identified modes:     
        if np.max(MAC[:, i]) >= ThMAC:                          # if the max MAC in this col is larger than the threshold
            j = np.argmax(MAC[:, i])                            # find the largest entry i.e. the best matching computed mode and
            modepairs = np.vstack([modepairs, [i + 1, j + 1]])  # write info to "modepairs": i: identified mode nr., j: corresponding model mode nr.

    # Make selection matrices to select pairs of modes
    L_id = np.zeros((Phi_id.shape[1], modepairs.shape[0]))      # preallocate selection matrix size
    L = np.zeros((Phi_m.shape[1], modepairs.shape[0]))
    for i in range(modepairs.shape[0]):                         
        L_id[modepairs[i, 0] - 1, i] = 1           # assemble mode selection matrix: identified modes
        L[modepairs[i, 1] - 1, i] = 1              # assemble mode selection matrix: modelled modes

    # Re-order / select modes and frequencies
    Phi_ids = Phi_id @ L_id                                                  # selected "re-ordered" identified modes
    freq_ids = np.take(freq_id, (modepairs[:, 0] - 1).astype(int)).tolist()  # and frequencies
    Phi_ms = Phi_m @ L                                                       # selected "re-ordered" modelled modes
    freq_ms = np.take(freq_m, (modepairs[:, 1] - 1).astype(int)).tolist()    # and frequencies

    return modepairs, Phi_ids, freq_ids, Phi_ms, freq_ms