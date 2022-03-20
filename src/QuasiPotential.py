#Import python modules for deterministic simulations
import sympy as sm
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import numpy as np
import math

#Import python module for gillespie simulations
import random

#Import python module for multiprocessing
import multiprocessing as mp

#Numerical Integration of the Waddington epigeneitc landscape
#Given an ode system for your genetic circuit, calculate the trajectoires alone the potential surface and then align them.

def intgrate_potential_path(odesys, 
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
    return x_list, v_list

def paths_2D_grid(xy_min:float, xy_max:float, xy_step:float, n_jobs:int, *params):
    """
    function for integrating trajectories on two dimensional grid grid

    Parameters:
    xy_min - minimum value of the grid of values
    xy_max - maximum value of the grid of values
    xy_step - step size acros the range of xy_values
    n_jobs - integer representing the number of cpus we are using analysis
    
    if n_jobs is -1, we use all the cpus available.

    Returns:
    paths - list of list of state lists and potential list for a grid of initial states
    """
    if n_jobs > 0:
        if n_jobs = 1:
            paths = [intgrate_potential_path([x,y], *params) for x in range(xy_min, xy_max, xy_step) for y in range(xy_min, xy_max, xy_step)]
        else:
            pool  = mp.Pool(n_jobs)
            paths = pool.startmap(integrate_potential_path, [([x,y], *params) for x in range(xy_min, xy_max, xy_step) for y in range(xy_min, xy_max, xy_step)])
            pool.close()
    else:
        cpus = mp.cpu_count()
        pool = mp.Pool(cpus)
        paths = pool.startmap(integrate_potential_path, [([x,y], *params) for x in range(xy_min, xy_max, xy_step) for y in range(xy_min, xy_max, xy_step)]) 
        pool.close()
    return paths

#To Do Path alignment and stochastic simulations
def path_alignment

