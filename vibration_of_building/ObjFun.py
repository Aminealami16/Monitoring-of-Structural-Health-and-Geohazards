def ObjFun(x, sclx1, sclx2):

    """
    Cost function, optimization and MAC matrices calculation
    
    Parameters:
    - x: vector containing variables for optimisation
    - sclx1: scale variable 1
    - sclx2: scale variable 2
    
    Returns:
    - f: Objective function value
    
    """
    
    # Import packages
    import numpy as np
    import pandas as pd
    import math
    import matplotlib.pyplot as plt
    
    from FE_fun import FE_fun
    from modematching import modematching
    
    # Load saved results from identification 
    loaded_data_npy = np.load('identified_eigdata.npy', allow_pickle=True).item()
    Phi_id = loaded_data_npy['Phi_id']
    freq_id = loaded_data_npy['freq_id']
    ind_d = loaded_data_npy['ind_d']
    
    # Calculate Phi and Omega for set of design variables using FE-model
    Phi, Omega = FE_fun((x[0]*sclx1, x[1]*sclx2))
    Phi_m = Phi[ind_d, :]
    Phi_m = Phi_m / np.linalg.norm(Phi_m, axis=0)
    freq_m = Omega/(2*np.pi)
    
    # Pair calculated modes with measured modes using MAC value:
    modepairs, Phi_ids, freq_ids, Phi_ms, freq_ms = modematching(Phi_id, freq_id, Phi_m, freq_m)
    nmatch = len(modepairs)     # Number of matched modes
    
    ## Compute cost function
    # Formulate a cost function and loop over identified modes/frequencies, use:  
    # - freq_ids(i)  : matched identified i^th frequency
    # - Phi_ids(:,i) : matched identified i^th mode  
    # - freq_s(i)    : matched computed i^th frequency
    # - Phi_s(:,i)   : matched computed i^th mode 
    #
    # - maxnomod     :specify maximum number of modes to be used 
    #                   * 1) can be used to limit the number of modes to less than the number of matched modes  
    #                   * 2) number of matched modes is the maximum number of usable modes
    #                   * 3) if number of matched modes reduces due to
    #                   parameter choise to value lower than "maxnomod", then
    #                   nmatch = len(modepairs) will be used as nomaxmod.
    #---------------------------------------------------------------------------------------------------------#
    
    # Define a maxium number of modes to be used
    maxnomod = ***
    #print(min(maxnomod, nmatch))
    
    T1 = np.zeros((min(maxnomod, nmatch),))        # First term of the cost function
    T2 = np.zeros((min(maxnomod, nmatch),))        # Second term of the cost function

    for i in range(min(maxnomod, nmatch)):         # for the number of matched modes or less
        
        # Normalise eigenmodes and ensure they have the same sign 
        Phi_ms[:,i] = Phi_ms[:,i] / np.linalg.norm(Phi_ms[:,i])      # Extract and normalize mode
        inprod = Phi_ids[:,i].T * Phi_ms[:,i]                        # Compute inner product with identified mode
        Phi_ms[:,i] = np.sign(inprod) * Phi_ms[:,i]                  # Switch sign if required

        # define a cost function 
        T1[i] = ***
        T2[i] = ***
    
    # The objective function is the summation of T1 and T2
    # Note: In order to keep the cost function smooth, it is important to account for a possibly varying number of modes used.
    f = ***
    
    return f
    
    
    
    