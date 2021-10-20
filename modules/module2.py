import numpy as np
from numpy import random
from numpy import linalg
from scipy import special as sp

####################################################################################################
########################               MAIN FUNCTIONS               ################################
####################################################################################################

def sherlock_func(N, delta):           #N is the number of particles in the beam
    '''
    Implementation of Sherlock's algorithm for particle-fluid collision [J. Comp. Phys. 2008]

        Parameters: 
            N (int): The number of particles in the beam
            delta (float): The time-step
        Returns:
            two arrays of time and velocities

    '''

    #initializing the beam of N particles
    (c_vec, u, vT, v_ell, Ac, Bc) = init_velocity()  
    # v_ell is the modulus of v_ell to normalize the final solution
    # c_vec is the particle velocity in fluid's frame;
    # vT is the background thermal speed; 
    # U is the fluid velocity
    # Ac dimension quantities needed in the function time_scales()
    # Bc ?

    (c0, Ainv) = c_quantities(c_vec)    
    # C0:   initial scalar velocity;
    # Ainv: inverse matrix
    
    vec = np.array([0.0, 0.0, c0])      
    # particle velocity in the fluid's frame + along the w_z-axis
    # (we rotate to z-component simply by v_z = c0)
    
    (tau_t, tau_parallel, tau_perp) = time_scales(G, c0, vT, Ac, Bc)  
    # Time scales computation
    
    tau_ch = min(tau_t, tau_parallel, tau_perp)  
    # we choose the minimum time scale as characteristic time
    
    a = 0.0  # time start ?
    b = 8.0  # time end ?
    dt = delta * tau_ch  
    # we select dt much shorter than tau_ch
    Ntime = int((b - a) / delta)
    # number of time steps

    # t = np.arange(a, b + 0.01, delta) 
    t = np.arange(a, b + delta, delta)      
    # grid time from a to b in delta steps
    vx_matrix = np.zeros((N, Ntime+1))
    vy_matrix = np.zeros((N, Ntime+1))
    vz_matrix = np.zeros((N, Ntime+1))
    # init. the matrix for each velocity component

    # We run a loop for a beam of N particles with the same initial conditions
    for i in range(N):          
        sol = time_evolution(Ntime, dt, vec, Ainv, u, v_ell, vT, Ac, Bc)
        (vx, vy, vz) = components_extraction(sol)
        vx_matrix[i] = vx
        vy_matrix[i] = vy
        vz_matrix[i] = vz

    vx_mean = average_values(vx_matrix)         #average vx
    vy_mean = average_values(vy_matrix)         #average vy
    vz_mean = average_values(vz_matrix)         #average vz
    v = norm(vx_mean, vy_mean, vz_mean)         #norm of the avergae speed
    
    return (t, v)

def sherlock_func_test(N, delta):           #N is the number of particles in the beam
    '''
    Implementation of Sherlock's algorithm for particle-fluid collision [J. Comp. Phys. 2008]

        Parameters: 
            N (int): The number of particles in the beam
            delta (float): The time-step
        Returns:
            two arrays of time and velocities

    '''
    (c_vec, u, vT, v_ell, Ac, Bc) = init_velocity()  
    (c0, Ainv) = c_quantities(c_vec)
    vec = np.array([0.0, 0.0, c0])        
    (tau_t, tau_parallel, tau_perp) = time_scales(G, c0, vT, Ac, Bc)   
    tau_ch = min(tau_t, tau_parallel, tau_perp)    
    a = 0.0  # time start ?
    b = 8.0  # time end ?
    dt = delta * tau_ch  
    Ntime = int((b - a) / delta)
    t = np.arange(a, b + delta, delta)

    vx_matrix = np.zeros((N, Ntime+1))
    vy_matrix = np.zeros((N, Ntime+1))
    vz_matrix = np.zeros((N, Ntime+1))
    for i in range(N):          
        sol, arr_tau_t, arr_tau_parallel, arr_tau_perp, arr_flim, arr_dt = time_evolution_test(Ntime, dt, vec, Ainv, u, v_ell, vT, Ac, Bc, c0)
        (vx, vy, vz) = components_extraction(sol)
        vx_matrix[i] = vx
        vy_matrix[i] = vy
        vz_matrix[i] = vz
    vx_mean = average_values(vx_matrix)         #average vx
    vy_mean = average_values(vy_matrix)         #average vy
    vz_mean = average_values(vz_matrix)         #average vz
    v = norm(vx_mean, vy_mean, vz_mean)         #norm of the avergae speed
    
    return (t, v, arr_tau_t, arr_tau_parallel, arr_tau_perp, arr_flim, arr_dt)

def changes_in_velocity(vec, dt, vT, Ac, Bc, G, v_parallel, v2_parallel, v2_perp):
    '''
    Calculation of the velocity change due to the particle-fluid collision

        Parameters:
            vec (array of float): particle velocity in the fluid frame along z-axis
            dt (float): timesteps
            vT (float): fluid thermal velocity
            Ac/Bc (float): dimensional parameters
            G (function): returns the G(w/vth) term in the slowing down equation (i.e., EQ.1)
            v_parallel (function): returns the term $\partial w_{\parallel} / \partial t / Ac$
            v2_parallel (function): returns the term $\partiial w_{\parallel}^2 / \partial t / Bc$
            v2_perp (function): returns the perpendicular term of the above EQ. 

        Returns:
            an updated velocity of particle in the fluid frame along the z-axis
    '''
    v = linalg.norm(vec)                                    #norm of the particle velocity in fluid's frame
    x = v/vT                                                #velocity ratio this is sqrt(x) = v/vT
    flim = limiting_factor(dt, G, v, vT, Ac, Bc)            #limiting factor computation
    ############## DETERMINISTIC EVOLUTION ##############
    dvz_det = dt * (Ac * v_parallel(G, x))
    ############# STANDARD DEVIATIONS COMPUTATION  ##############
    sigma_parallel = np.sqrt(dt * Bc * abs(v2_parallel(G, x)))
    sigma_perp = np.sqrt(dt * Bc * abs(v2_perp(G, x)))        # here the abs() is a must since it is under the sqrt().
    ########## COMPUTATION OF THE STOCHASTIC CHANGES ###########
    dvz_sto = N(sigma_parallel)                             #z-component stochastic change
    v_perp = N(sigma_perp)                                  #random velocity modulus in diffusion process
    angle_perp = theta()                                    #random scattering angle in diffusion process
    dvx = v_perp * np.cos(angle_perp)                       #x-component stochastic change
    dvy = v_perp * np.sin(angle_perp)                       #y-component stochastic change
    dvz = flim*dvz_det + dvz_sto                          #total change in z-component
    # dvz = flim*dvz_det                                      # @yaowp: dvz_sto is 100 times larger than dvz_det and dvxy
    dv = np.array([dvx, dvy, dvz])
    ########## ADDING THE VELOCITY CHANGES
    vec = vec + dv
    return (vec, dt)                        #we return a vector that contains the changes in velocity

def time_evolution(Ntime, dt, vec, Ainv, u, v_ell, vT, Ac, Bc):
    '''
    This function evolves in time the velocity

        Parameters:
            Ntime (int): number of timesteps
            dt (float): timesteps
            vec (array of float): velocity in three directions
            Ainv (2darray of float): reverse transformation matrix from w_z-axis back to the fluid frame
            u (array of float): fluid bulk velocity in three directions
            v_ell (array of float): particle velocity in the lab frame in three directions
            vT (float): thermal velocity of the fluid
            Ac: for dimensional purpose [m/s^2]
            Bc: for dimensional purpose [m^2/s^3]

        Returns:
            sol (array of float): Number of time steps, particle velocity (x, y, z)
    '''
    # 
    sol = np.zeros((Ntime+1, 3))                # we save the velocity after solving the problem
    sol[0] = vec                                # we save the initial velocity
    #from here we are in the fluid's frame of reference
    # we loop over all timesteps to get each particle velocity change due to collision in the fluid frame along z-axis 
    for i in range(Ntime):
        (sol[i + 1], dt) = changes_in_velocity(sol[i], dt, vT, Ac, Bc, G, v_parallel, v2_parallel, v2_perp) 
        # it returns the new velocity
    sol = matrix_resul(Ainv, sol)               # rotation to the fluid velocity NOT along the z-axis
    sol = translation(sol, u)                   # change from the fluid velocity to the lab frame 
    sol = sol/linalg.norm(v_ell)                # normalization of the total solution
    return sol

def time_evolution_test(Ntime, dt, vec, Ainv, u, v_ell, vT, Ac, Bc, c0):
    '''
    This function evolves in time the velocity
    '''
    sol = np.zeros((Ntime+1, 3))                # we save the velocity after solving the problem
    sol[0] = vec                                # we save the initial velocity
    arr_tau_t        = np.zeros(Ntime+1)
    arr_tau_parallel = np.zeros(Ntime+1)
    arr_tau_perp     = np.zeros(Ntime+1)
    arr_flim         = np.zeros(Ntime+1)
    arr_dt           = np.zeros(Ntime+1)

    (arr_tau_t[0], arr_tau_parallel[0], arr_tau_perp[0]) = time_scales(G, c0, vT, Ac, Bc) 
    arr_flim[0] = 1.0
    arr_dt[0] = dt

    #from here we are in the fluid's frame of reference
    # we loop over all timesteps to get each particle velocity change due to collision in the fluid frame along z-axis 
    for i in range(Ntime):
        (sol[i + 1], arr_dt[i+1], arr_tau_t[i+1], arr_tau_parallel[i+1], arr_tau_perp[i+1], arr_flim[i+1]) = changes_in_velocity_test(sol[i], arr_dt[i], vT, Ac, Bc, G, v_parallel, v2_parallel, v2_perp) 

        # it returns the new velocity
    sol = matrix_resul(Ainv, sol)               # rotation to the fluid velocity NOT along the z-axis
    sol = translation(sol, u)                   # change from the fluid velocity to the lab frame 
    sol = sol/linalg.norm(v_ell)                # normalization of the total solution
    return sol, arr_tau_t, arr_tau_parallel, arr_tau_perp, arr_flim, arr_dt

def changes_in_velocity_test(vec, dt, vT, Ac, Bc, G, v_parallel, v2_parallel, v2_perp):
    '''
    Calculation of the velocity change due to the particle-fluid collision
    '''
    v = linalg.norm(vec)                                    #norm of the particle velocity in fluid's frame
    x = v/vT                                                #velocity ratio this is sqrt(x) = v/vT
    (flim, tau_t, tau_parallel, tau_perp) = limiting_factor_test(dt, G, v, vT, Ac, Bc)            #limiting factor computation
    ############## DETERMINISTIC EVOLUTION ##############
    dvz_det = dt * (Ac * v_parallel(G, x))
    ############# STANDARD DEVIATIONS COMPUTATION  ##############
    sigma_parallel = np.sqrt(dt * Bc * abs(v2_parallel(G, x)))
    sigma_perp = np.sqrt(dt * Bc * abs(v2_perp(G, x)))        # here the abs() is a must since it is under the sqrt().
    ########## COMPUTATION OF THE STOCHASTIC CHANGES ###########
    dvz_sto = N(sigma_parallel)                             #z-component stochastic change
    v_perp = N(sigma_perp)                                  #random velocity modulus in diffusion process
    angle_perp = theta()                                    #random scattering angle in diffusion process
    dvx = v_perp * np.cos(angle_perp)                       #x-component stochastic change
    dvy = v_perp * np.sin(angle_perp)                       #y-component stochastic change
    dvz = flim*dvz_det + dvz_sto                          #total change in z-component
    # dvz = flim*dvz_det                                      # @yaowp: dvz_sto is 100 times larger than dvz_det and dvxy
    dv = np.array([dvx, dvy, dvz])
    ########## ADDING THE VELOCITY CHANGES
    vec = vec + dv
    return (vec, dt, tau_t, tau_parallel, tau_perp, flim)      

def parameters():
    '''
    Set the parameters of the particle and fluid in SI units

        Returns:
            vT: thermal speed of the fluid
            gamma_tb: ???
            Ac: ???
            Bc: ???
    '''
    Zb = 1                                  #Z for background, we set an electron-fluid background
    Z = 1                                  #Z for ions, we use Aluminum ions
    A = 27                                  #mass number for ions, we use alumnium
    me = 9.10938291e-31                     #electron mass
    mp = 1.672621898e-27                    #proton mass
    mi = A*mp                               #ions mass
    nb = 7e23                               #ion particle density [m^-3 ???]
    # nb = 7e22
    q_elec = 1.602176565e-19                #electron charge
    qt = Z*q_elec                           #test particle charge, we consider an electron
    # qb = -Zb*qt                             #background ion-particle charges ???
    qb = -Zb*q_elec                         # background is electron
    lnCol = 17.0                            #Coulomb logarithm experimental value for laboratory plasma
    # lnCol = 3.0
    e_0 = 8.8541878176204e-12               #vacuum permittivity
    eV = 1.60217733e-19                     #eV to Jules
    kT = 1.0                                #electron temperature times Boltzmann constant
    # mu = 8.0e7                              # @yaowp: ???
    mu = mi/me                              # test particle Al ion and fluid electron ???

    gamma_tb = (qt**2.0 * qb**2.0 * nb * lnCol)/(2.0 * np.pi * e_0**2.0 * mi**2.0)  #Gamma_tb factor
    vT = np.sqrt(2.0*kT*eV/me)                                                     #thermal speed
    #Ac and Bc are the factors to dimensionless the coefficients
    Ac = (gamma_tb / vT**2.0) * (1.0 + mu)            #Ac has acceleration dimensions
    Bc = gamma_tb / vT                              #Bc has dimensions of velocity square over second
    return (vT, gamma_tb, Ac, Bc)

#### SPECIAL FUNCTION DEFINITION ########
def G(x):
    return x / (2.0*(x**3.0) + (3.0/(2.0 * np.sqrt(np.pi))))  # Sherlock's approx

#### DIMENSIONLESS COEFFICIENTS DEFINITION #####
def v_parallel(G, x):               # frictional coeff
    # print('v_parallel = {:.2e}'.format(-G(x)))
    return -G(x)
def v2_parallel(G, x):              # parallel diffusion coeff
    # print('v2_parallel = {:.2e}'.format(G(x)/x))
    return G(x)/x
def v2_perp(G, x):                  # perpendicular diffusion coeff
    # print('v2_perp = {:.2e}'.format((sp.erf(x) - G(x))/x))
    return (sp.erf(x) - G(x))/x

#### random functions #####
def N(sigma): 
    # we generate a random number with a centered normal distribution with standard deviation sigma
    return random.normal(0.0, sigma)

def theta():
    return 2.0*np.pi*random.uniform(0.0, 1.0)

#### rotation matrix ###
def rot_mat(th, ph):  
    #rotation matrix
    A = np.array([[ np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)],
                  [-np.sin(ph),            np.cos(ph),             0.0],
                  [ np.cos(ph)*np.sin(th), np.sin(th)*np.sin(ph),  np.cos(th)]
                  ])
    return A

def inv_rot_mat(th, ph):   
    '''
    inverse rotation matrix from pre-calculated angles using EQ.5 & EQ.6
    '''
    A = np.array([[ np.cos(th)*np.cos(ph), -np.sin(ph), np.cos(ph)*np.sin(th)],
                  [ np.cos(th)*np.sin(ph),  np.cos(ph), np.sin(th)*np.sin(ph)],
                  [-np.sin(th),             0.0,        np.cos(th)]
                  ])
    return A

### c computation ###
def c_quantities(c_vec):
    '''
    Calculate the transformation matrix from the fluid frame to the w_z-axis frame

        Parameters:
            c_vec (array of float): particle velocities in the fluid frame

        Returns:
            c (float): np.sqrt(c_x^2 + c_y^2 + c_z^2) [m/s]
            Ainv (2darray of float): reverse transformation matrix for later use
    '''
    c = linalg.norm(c_vec)  #norm
    c_perp = np.sqrt(c_vec[0] ** 2.0 + c_vec[1] ** 2.0)     #norm in the perpendicular plane, EQ.7
    ####we consider the possible values for angle theta
    th = np.arccos(c_vec[2] / c)  # EQ.5
    ph = np.arccos(c_vec[0] / c_perp)  # EQ.6
    Ainv = inv_rot_mat(th, ph) # inv rot matrix
    return (c, Ainv) # it returns the norm and the inverse matrix to come back to the lab frame

#### SCALE TIMES ####
def time_scales(G, v, vT, Ac, Bc):
    '''
    Calculate the characteristic time scales for the limiting factor ($f_{lim}$) in EQ.12

        Parameters:
            G (?): ???
            v (float): particle velocity
            vT (float): fluid thermal velocity
            Ac: ??? the dimension quantities
            Bc: ??? the dimension quantities

        Returns:
            tau_t (float): the slowing time in EQ.13
            tau_parallel (float): the scattering time in the parallel direction in EQ.14
            tau_perp (float): the scattering time in the perpendicular direction in EQ.15

        Notes:
            If we use the approximations for special functions, 
            you can get negative values and it's necessary to take the abs value ???

    '''
    x = v/vT
    tau_t = v / (Ac*abs(v_parallel(G, x)))                      #slowing characteristic time
    tau_parallel = v**2.0 / (Bc*abs(v2_parallel(G, x)))         #parallel diffusion characteristic time
    tau_perp = v**2.0 / (Bc*abs(v2_perp(G, x)))                 #perpendicular diffusion characteristic time
    # print('tau_t = {:.2e}'.format(tau_t))
    # print('tau_parallel = {:.2e}'.format(tau_parallel))
    # print('tau_perp = {:.2e}'.format(tau_perp))
    if(min(tau_t,tau_parallel,tau_perp) - tau_t >= 1.0e-20):
        print('tau_t is not the min.')
    return (tau_t, tau_parallel, tau_perp)

def init_velocity():
    '''
    Initialization of the particle velocities

        Returns:
            c_vec: particle velocities in the fluid frame
            u: fluid bulk velocity
            vT: fluid thermal velocity [from parameters()] 
            v_ell: particle velocities in the lab frame
            Ac: ??? [from parameters()] 
            Bc: ??? [from parameters()] 
    '''
    (vT, gamma_tb, Ac, Bc) = parameters()
    # u = 10**5 * np.array([1.0 / 3.0, -1.0 / 2.0, -1.0 / 3.0])       #fluid velocity in lab frame
    # v_ell = 10**5 * np.array([4.0 / 3.0, 1.0 / 2.0, 1.0 / 6.0])     #particle velocity in lab frame
    u = np.array([0.0, 0.0, 0.0])                                  # set the fluid bulk velocity in the lab frame
    v_ell = 1.0e5 * np.array([1.0, 1.0, 0.5])                      # set the particle velocity in the lab frame
    # v_ell = 1.0e5 * np.array([1.0, 1.0, 1.0]) 
    c_vec = v_ell - u                                              # particle velocity in fluid's frame
    return(c_vec, u, vT, v_ell, Ac, Bc)

def limiting_factor(dt, G, v, vT, Ac, Bc):
    '''
    Ensure that the time-step will always be less than the relevant relaxation time for every particle
    '''
    (tau_t, tau_parallel, tau_perp) = time_scales(G, v, vT, Ac, Bc)
    flim = min(1.0, tau_t / (2.0 * dt), tau_parallel / (2.0 * dt), tau_perp / (2.0 * dt))
    # print('tau_t / 2dt = {:.2e}'.format(tau_t / (2.0 * dt)))
    # print('tau_parallel / 2dt = {:.2e}'.format(tau_parallel / (2.0 * dt)))
    # print('tau_perp / 2dt = {:.2e}'.format(tau_perp / (2.0 * dt)))
    # print('flim = {:.2e}'.format(flim))
    # if((1.0 - flim) > 0.0): 
    #     print('flim is smaller than 1.0')
    return flim

def limiting_factor_test(dt, G, v, vT, Ac, Bc):
    '''
    Ensure that the time-step will always be less than the relevant relaxation time for every particle
    '''
    (tau_t, tau_parallel, tau_perp) = time_scales(G, v, vT, Ac, Bc)
    flim = min(1.0, tau_t / (2.0 * dt), tau_parallel / (2.0 * dt), tau_perp / (2.0 * dt))
    # print('tau_t / 2dt = {:.2e}'.format(tau_t / (2.0 * dt)))
    # print('tau_parallel / 2dt = {:.2e}'.format(tau_parallel / (2.0 * dt)))
    # print('tau_perp / 2dt = {:.2e}'.format(tau_perp / (2.0 * dt)))
    # print('flim = {:.2e}'.format(flim))
    # if((1.0 - flim) > 0.0): 
    #     print('flim is smaller than 1.0')
    return  (flim, tau_t, tau_parallel, tau_perp)

def components_extraction(solc):
    # this function extract the velocity components from the total solution
    solc = solc.transpose()
    vx = solc[0]
    vy = solc[1]
    vz = solc[2]
    return (vx, vy, vz)

#SUMPLEMENTARY FUNCTIONS

def average_values(M):
    #With this function we compute the average velocity of all the ensemble
    # @yaowp: lib instead ???
    M = M.transpose()
    vxt = np.array([])
    for i in range(len(M)):
        vxt = np.append(vxt, np.mean(M[i]))
    return vxt

def norm(vx, vy, vz):
    #we compute the norm of the average velocity
    aux = np.array([])
    for i in range(len(vx)):
        aux = np.append(aux, np.sqrt(vx[i]**2.0+ vy[i]**2.0 + vz[i]**2.0))
    return aux

def matrix_resul(A, B): 
    '''
    This function multiplies the final solution with the inverse rotation matrix

        Parameters: 
            A (2darray of float): the inverse rotation matrix
            B (1darray of float): velocity in the fluid's frame in three directions

        Note:
            Are there any lib for matrix multiple rather than this for loop ???
    '''
    for i in range(len(B)):
        B[i] = np.dot(A, B[i])
    return B

def norma_matrix(sol):
    #This function compute the norm of each velocity vector
    aux = np.array([])
    for i in range(len(sol)):
        aux = np.append(aux, linalg.norm(sol[i]))
    return aux

def translation(M, u):
    '''
    Transform the particle velocity in the fluid frame back into the lab frame, i.e., v = c + u

        Parameters:
            M (array of float): particle velocity in the fluid frame
            u (array of float): fluid velocity in the lab frame

        Returns:
            M (array of float): particle velocity in the lab frame

        Notes: 
            a loop over the number of timesteps (better way to avid the loop?)
    '''
    for i in range(len(M)):
        M[i] = M[i] + u
    return M


def delta_Ti(vT, Ti, Te):
    Z = 13
    A = 27
    q_elec = 1.602176565e-19
    nb = 7e23
    lnCol = 17.0
    me = 9.10938291e-31
    mp = 1.672621898e-27
    mi = A * mp
    nu_e = (16.0*np.sqrt(np.pi)*(Z**2.0)*(q_elec**4.0)*nb*lnCol)/(me*mi*vT**3.0)
    dTi = -(2.0/3.0)*nu_e*(Ti - Te)
    return (dTi, 1.0/nu_e)

def temperature_equilibrium(Ntime, delta):
    eV = 1.60217733e-19
    Te = 1.0*eV
    Ti = 10*Te
    Ti0 = Ti
    me = 9.10938291e-31
    mp = 1.672621898e-27
    A = 23
    mi = A*mp
    vT = np.sqrt(2.0*(Ti/mi + Te/me))
    (dT0, tau_e) = delta_Ti(vT, Ti, Te)
    dt = delta*tau_e
    T = np.array([])
    T = np.append(T, Ti0/Ti0)
    for i in range(Ntime):
        (dT, tau_e) = delta_Ti(vT, Ti, Te)
        Ti = Ti + dt*dT
        vT = np.sqrt(2.0 * (Ti / mi + Te / me))
        T = np.append(T, Ti / Ti0)
    return T

def sherlock_funcT(N, delta):           #N is the number of particles in the beam
    a = 0.0                     #lower time limit
    b = 8.0                     #upper time limit
    Ntime = int((b-a)/delta)    #number of  points to simulate the time grid
    t = np.arange(a, b + 0.01, delta)
    T_matrix = np.zeros((N, Ntime+1))
    for i in range(N):          #We run a loop for a beam of N particles with the same initial conditions
        TN = temperature_equilibrium(Ntime, delta)
        T_matrix[i] = TN
    T = average_values(T_matrix)         #average vx
    return (t, T)
