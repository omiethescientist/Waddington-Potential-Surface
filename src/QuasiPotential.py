#Import python modules for deterministic simulations
import sympy as sm
from scipy.integrate import odeint
from scipy.spatial.distance import euclidean
import numpy as np
import math

#Import python module for multiprocessing
import multiprocessing as mp

#Numerical Integration of the Waddington epigeneitc landscape
#Given an ode system for your genetic circuit, calculate the trajectoires alone the potential surface and then align them.

def integrate_potential_path(odesys, 
        x0, 
        t0:float, 
        V0:float, 
        tol:float, 
        tstep:float,
        *params,
        ):
    """
    function for integrating trajectories on the potential function
    
    Parameters:
    odesys - function that represents the system of differential equations 
    
    e.g. 
    def dXdt(x,t,*p):
        fold_yx, fold_xy,  k_yx, k_xy, b_x, b_y, d_x, d_y, n = p
        return  b_x + fold_yx*k_yx**n/(k_yx**n + y**n) - d_x*x

    def dYdt(y,t,*p):
        fold_yx, fold_xy,  k_yx, k_xy, b_x, b_y, d_x, d_y, n = p
        return b_y + fold_xy*k_xy**n/(k_xy**n + x**n) - d_y*y

    def double_neg(z0, t, *p):
        x,y = z0
        del_x = dxdt(x,y,*p)
        del_y = dydt(x,y,*p)
        return [del_x, del_y]

    x0 - list of lists - initial state of the system
    t0 - float - initial time 
    V0 - initial potential
    tol - error tolerance for convergence
    tstep - size of the time steps for integration
    
    Returns:
    x_list - list of lists of states of the system at each time point
    v_list - list of potential values at each state of the system along its trajectory on the phase plane
    """
    #set convergence criterion
    del_dist = np.sqrt(2*tol**2)

    #Set initial states
    x_list = [x0]
    v_list = [V0]
    t_list = [t0]
    
    #While loop for integration
    while del_dist >= tol:
        #Get t-1 values
        x = np.array(x_list[-1])
        V = v_list[-1]
        t = t_list[-1]

        #Integrate the next time step
        dx_dt = odesys(x, t, *params)
        xvals = odeint(odesys, x, [t, t+tstep], params)
        del_x = xvals[1] - xvals[0]
        
        #Calculate new distance from previous state
        del_dist = np.sqrt(sum(del_x**2))
        
        #Calulate change in V
        del_V = -(np.sum(dx_dt * del_x))
        
        #Calculate new values 
        V = V + del_V
        x = x + del_x
        t = t + tstep

        #Add values to the lists
        x_list.append(x.tolist())
        v_list.append(V)
        t_list.append(t)
    return x_list, v_list, t_list

def QuasiPotential2D(odesys, ode_params, integration_params, xy_min:float, xy_max:float, xy_step:float, attractor_tol: float = 1e-6, n_jobs:int = 4):
    """
    function for integrating trajectories on two dimensional grid 
    then align them based on their attractors and initial states

    Parameters:
    odesys - function that represents the system of ordinary differential equations
    ode_params - parameters for the odesys
    integration_params - parameters for integrating the potential path via the integrate_potential_path function
    xy_min - minimum value of the grid of values
    xy_max - maximum value of the grid of values
    xy_step - step size acros the range of xy_values
    n_jobs - integer representing the number of cpus we are using analysis
    
    if n_jobs is -1, we use all the cpus available.

    Returns:
    x_paths - X Coordinates of the Quasipotential Surface
    y_paths - Y Coordinates of the Quasipotential Surface
    v_paths - Potential Coordinates of the Quasipotenital Surface
    attractors - Return attractors found on the quasipotential surface at grid coordinates
    """
    # Based on whether or not we are using all or some of the processors
    if n_jobs > 0:
        #Run serially if using 1
        if n_jobs == 1:
            paths = [integrate_potential_path(odesys,[x,y], *integration_params, *ode_params) for x in np.arange(xy_min, xy_max, xy_step) for y in np.arange(xy_min, xy_max, xy_step)]
        #Run in parallel
        else:
            pool  = mp.Pool(n_jobs)
            paths = pool.starmap(integrate_potential_path, [(odesys, [x,y], *integration_params, *ode_params) for x in np.arange(xy_min, xy_max, xy_step) for y in np.arange(xy_min, xy_max, xy_step)])
            pool.close()
    #Run in parallel with all processors
    else:
        cpus = mp.cpu_count()
        pool = mp.Pool(cpus)
        paths = pool.starmap(integrate_potential_path, [(odesys, [x,y], *integration_params, *ode_params) for x in np.arange(xy_min, xy_max, xy_step) for y in np.arange(xy_min, xy_max, xy_step)]) 
        pool.close()
    
    #Alignment of path potentials
    attractors = []
    initial_states = []
    x_paths = []
    y_paths = []
    v_paths = []
    for p in range(len(paths)):
        path = paths[p]
        
        # Add the X path and Y path as they don't need changing
        X, Y = zip(*path[0])
        x_paths.append(X)
        y_paths.append(Y)
        
        V = path[1]
        initial_state = (X[0], Y[0],V[0])
        attractor = (X[-1], Y[-1], V[-1])

        #If this is the first path
        if len(attractors) == 0:
            attractors.append(attractor)
            initial_states.append(initial_state)
            v_paths.append(V)
        else:
            dists = [euclidean(attractor[:2], attractor_prev[:2]) for attractor_prev in attractors]
            dist_bools = [dist <= attractor_tol for dist in dists]
            #If there is a similar attractor
            if sum(dist_bools) > 0:
                attractor_prev = attractors[dist_bools.index(True)]
                v_path = [v - (attractor[2] - attractor_prev[2]) for v in V]
                v_paths.append(v_path)
                initial_states.append(initial_state)
            #If there is no similar attractor, align to the previous initial state
            else:
                dists = [euclidean(initial_state[:2], initial_state_prev[:2]) for initial_state_prev in initial_states]
                dist_bools = [dist <= np.sqrt(2*xy_step**2) for dist in dists]
                initial_state_prev = initial_states[dist_bools.index(True)]
                v_path = [v - (initial_state[2] - initial_state_prev[2]) for v in V]
                v_paths.append(v_path)
                attractors.append(attractor)
    return x_paths, y_paths, v_paths, attractors
