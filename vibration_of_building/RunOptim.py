def RunOptim():
    
    """
    Brute force assessment and gradient based optimization
    
    """
    
    # Import packages
    import numpy as np
    import pandas as pd
    import math
    import matplotlib.pyplot as plt
    
    from scipy.optimize import minimize
    
    #Define variables and axes names
    x1 = ***                # Define the parameter space for each variable 
    x2 = ***
    x1txt = '***'           # Variable name for axis labels  
    x2txt = '***'               
    
    # Define scale factors
    sclx1 = ***  
    sclx2 = *** 
    
    # Scale variables - these are scaled back inside ObjFun.py!
    x1 = x1/sclx1
    x2 = x2/sclx2
    
    # Brute force assessment
    #--------------------- #
    from ObjFun import ObjFun
  
    # Evaluate the objective function on grid
    f = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            f[i, j] = ObjFun([x1[i], x2[j]], sclx1, sclx2)
    
    # Gradient based optimization 
    #-------------------------- #
    
    # Setup optimization: set bounds and initial guess for optimization variables
    b1 = [x1[0], x1[-1]]          # b1: bounds variable 1 - scaled
    b2 = [x2[0], x2[-1]]          # b2: bounds variable 2 - scaled
    x0 = [***, ***]               # x0: initial guess - scaled

    # Set output function
    def outfun(x):
        # During iteration:
        history['x'].append(x)
    
    # Set up shared variables
    history = {'x': []}
    
    # Set options for optimization
    options = {'disp': True}
    
    # Perform optimization
    result = minimize(ObjFun, x0, args=(sclx1, sclx2), method='Nelder-Mead', bounds=[b1, b2], options=options, callback=outfun)
    
    x_optimal = result.x
    f_optimal = result.fun
    history['x'] = np.array(history['x'])
    
    x1_optimal = x_optimal[0]*sclx1
    x2_optimal = x_optimal[1]*sclx2

    print(f'The optimal value of parameter 1 is: {x1txt} = {x1_optimal}') # optimal value parameter 1
    print(f'The optimal value of parameter 2 is: {x2txt} = {x2_optimal}') # optimal value parameter 2
    
    # plot the results in generated contour and surface plot
    #----------------------------------------------------- #
    
    # contour plot 
    #----------- #
    
    plt.figure(figsize=(12,8))
    plt.contour(x1*sclx1 , x2*sclx2 , f.T)
    plt.gca().set_facecolor((1, 1, 1))  
      
    # Scatter plot for final solution
    plt.plot(x_optimal[0]*sclx1, x_optimal[1]*sclx2, 'r*', markersize=12, label="Final solution")
   
    # Scatter plot for intermediate solutions
    plt.plot(x0[0]*sclx1, x0[1]*sclx2, 'g')
    plt.plot(history['x'][:, 0]*sclx1, history['x'][:, 1]*sclx2, 'go', label="Intermediate solutions")
    
    plt.title('Contour plot of objective function')
    plt.xlabel(f"{x1txt}")
    plt.ylabel(f"{x2txt}")
    plt.legend()
    
    # surface plot 
    #----------- #
    
    # Create meshgrid
    X, Y = np.meshgrid(x1*sclx1, x2*sclx2)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, f.T, cmap='viridis')
    
    # Plot final solution
    ax.scatter(x_optimal[0]*sclx1, x_optimal[1]*sclx2, f_optimal, color='red', marker='D', s=100, label='Final solution')
    ax.text(x_optimal[0]*sclx1, x_optimal[1]*sclx2, f_optimal, f'({x_optimal[0]*sclx1:.2f}, {x_optimal[1]*sclx2:.2f}, {f_optimal:.2f})', 
            color='k', fontsize=10, ha='right', va='bottom',zorder=20)

    ax.set_zlabel('Objective function value')
    ax.set_xlabel(f'{x1txt}')
    ax.set_ylabel(f'{x2txt}')
    ax.set_title('Surface plot of objective function')
    ax.legend()
    ax.view_init(elev=25, azim=60)
    plt.show()
    
    return x1_optimal, x2_optimal

    
    
    
    