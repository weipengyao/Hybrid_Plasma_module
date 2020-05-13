import numpy as np
from numpy import random
from numpy import linalg
from scipy import special as sp

####################################################################################################
########################               MAIN FUNCTIONS               ################################
####################################################################################################

def parameters():
    Z = 1
    Zb = 1
    me = 9.10938291e-31                     #electron mass
    mp = 1.672621898e-27                    #proton mass
    mu = me/me                              #electron-proton mass ratio
    nb = 7e23                               #ion particle density
    q_elec = 1.602176565e-19                #electron charge
    qt = Z*q_elec                           #test particle charge, we consider an electron
    qb = -Zb*qt                             #background ion-particle charges
    lnCol = 17.0                            #Coulomb logarithm experimental value for laboratory plasma
    e_0 = 8.8541878176204e-12               #vacuum permittivity
    eV = 1.60217733e-19                     #eV to Jules
    kT = 1.0                                  #electron temperature times Boltzmann constant

    gamma_tb = (qt**2.0 * qb**2.0 * nb * lnCol)/(2.0 * np.pi * e_0**2.0 * me**2.0)  #Gamma_tb factor
    vT = np.sqrt(2.0*kT*eV/me)                                                     #thermal speed
    # Ac and Bc are the factors to dimensionless the coefficients
    Ac = (gamma_tb / (vT**2.0)) * (1.0 + mu)            #Ac has acceleration dimensions
    Bc = gamma_tb / vT                              #Bc has dimensions of velocity square over second
    return (vT, gamma_tb, Ac, Bc)

#### SPECIAL FUNCTION DEFINITION ########
def G(x):
    #derf = (2.0 / np.sqrt(np.pi))*np.exp(-x**2.0) #error function derivative
    #return (sp.erf(x) - x*derf)/(2.0*x**2.0)
    return x / (2.0*x**3.0 + (3.0/(2.0 * np.sqrt(np.pi))))  #Sherlock's approx
#### DIMENSIONLESS COEFFICIENTS DEFINITION #####
def v_parallel(G, x):               #frictional coeff
    return -G(x)
def v2_parallel(G, x):              #parallel diffusion coeff
    return G(x)/x
def v2_perp(G, x):                  #perpendicular diffusion coeff
    return (sp.erf(x) - G(x))/x

#### random functions #####
def N(sigma): #we generate a random number with a centered normal distribution with standard deviation sigma
    return random.normal(0.0, sigma)
def theta():
    return 2.0*np.pi*random.uniform(0.0, 1.0)

#### rotation matrix ###
def rot_mat(th, ph):  #rotation matrix
    A = np.array([[np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)],
                  [-np.sin(ph), np.cos(ph), 0.0],
                  [np.cos(ph)*np.sin(th), np.sin(th)*np.sin(ph), np.cos(th)]])
    return A
def inv_rot_mat(th, ph):   #inverse rotation matrix
    A = np.array([[np.cos(th)*np.cos(ph), -np.sin(ph), np.cos(ph)*np.sin(th)],
                  [np.cos(th)*np.sin(ph), np.cos(ph), np.sin(th)*np.sin(ph)],
                  [-np.sin(th), 0.0, np.cos(th)]])
    return A

### c computation ###
def c_quantities(c_vec):
    c = linalg.norm(c_vec)  #norm
    c_perp = np.sqrt(c_vec[0] ** 2.0 + c_vec[1] ** 2.0)     #norm in the perpendicular plane
    ####we consider the possible values for angle theta
    th = np.arccos(c_vec[2] / c)
    ph = np.arccos(c_vec[0] / c_perp)
    '''
    if c_vec[2]>=0:
        th = np.arccos(c_vec[2] / c)                        #theta angle for the inverse matrix
    else:
        th = np.pi - np.arccos(abs(c_vec[2]) / c)

    ####we consider the possible values for angle phi
    if c_vec[0] >= 0 and c_vec[1]>=0:
        ph = np.arccos(c_vec[0] / c_perp)                   #phi angle for the inverse matrix
    elif c_vec[0] < 0 and c_vec[1]<0:
        ph = 2.0*np.pi - np.arccos(c_vec[0] / c_perp)
    elif c_vec[0] <= 0 and c_vec[1]>=0:
        ph = np.pi - np.arccos(abs(c_vec[0]) / c_perp)
    else:
        ph = 2.0*np.pi - np.arccos(abs(-c_vec[0]) / c_perp)
    '''
    Ainv = inv_rot_mat(th, ph) #inv rot matrix
    return (c, Ainv) #it returns the norm and the inverse matrix to come back to the lab frame

#### SCALE TIMES ####
def time_scales(G, v, vT, Ac, Bc):
    #Ac and Bc are the dimension quantities
    #If we use the approximations for special functions, you can get negative values and it's necessary to take the abs value
    x = v/vT
    tau_t = v / (Ac*abs(v_parallel(G, x)))                      #slowing characteristic time
    tau_parallel = v**2.0 / (Bc*abs(v2_parallel(G, x)))         #parallel diffusion characteristic time
    tau_perp = v**2.0 / (Bc*abs(v2_perp(G, x)))                 #perpendicular diffusion characteristic time
    return (tau_t, tau_parallel, tau_perp)

def matrix_resul(A, B): #this function multiplies the final solution with the inverse rotation matrix
    #A is the inverse rotation matrix
    #B is a matrix that contains the velocity vectors (solution).
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
    #This function add the fluid velocity to each velocity solution at the end
    #M is a matrix with solution
    #u is the fluid velocity in the lab frame
    for i in range(len(M)):
        M[i] = M[i] + u
    return M

def init_velocity():
    #here we initialise the velocities
    (vT, gamma_tb, Ac, Bc) = parameters()
    u = 1.0e5 * np.array([1.0 / 3.0, -3.0 / 6.0, -1.0 / 3.0])
    v_ell = 1.0e5 * np.array([4.0 / 3.0, 3.0 / 6.0, 1.0 / 6.0])
    c_vec = v_ell - u #particle velocity in fluid's frame
    return(c_vec, u, vT, v_ell, Ac, Bc)

def velocity_changes(vec, dt, x, v, vT, Ac, Bc, G, v_parallel, v2_parallel, v2_perp):
    ############## DETERMINISTIC EVOLUTION ##############
    dvz_det = dt * Ac * v_parallel(G, x)
    ############# STANDARD DEVIATIONS COMPUTATION  ##############
    sigma_parallel = np.sqrt(dt * Bc * abs(v2_parallel(G, x)))
    sigma_perp = np.sqrt(dt * Bc * abs(v2_perp(G, x)))
    ########## COMPUTATION OF THE STOCHASTIC CHANGES ###########
    dvz_sto = N(sigma_parallel)                             #z-component stochastic change
    v_perp = N(sigma_perp)                                  #random velocity modulus in diffusion process
    angle_perp = theta()                                    #random scattering angle in diffusion process
    dvx = v_perp * np.cos(angle_perp)                       #x-component stochastic change
    dvy = v_perp * np.sin(angle_perp)                       #y-component stochastic change
    flim = limiting_factor(dt, G, v, vT, Ac, Bc)            #limiting factor computation
    dvz = flim*(dvz_det) + dvz_sto                          #total change in z-component
    dv = np.array([dvx, dvy, dvz])
    ########## ADDING THE VELOCITY CHANGES
    vec = vec + dv
    return vec                        #we return a vector that contains the changes in velocity

def limiting_factor(dt, G, v, vT, Ac, Bc):
    (tau_t, tau_parallel, tau_perp) = time_scales(G, v, vT, Ac, Bc)
    flim = min(1.0, tau_t / (2.0 * dt), tau_parallel / (2.0 * dt), tau_perp / (2.0 * dt))
    return flim

def denergy(E0, v):
    E = np.array([])
    mp = 1.672621898e-27
    for i in range(len(v)):
        E = np.append(E, abs(0.5*mp*v[i]**2.0 - E0)/E0)
    return E

def components_extraction(solc):
    #this function extact the velocity components from the total solution
    vx = np.array([])
    vy = np.array([])
    vz = np.array([])
    for i in range(len(solc)):
        vx = np.append(vx, (solc[i, 0]))
        vy = np.append(vy, (solc[i, 1]))
        vz = np.append(vz, (solc[i, 2]))
    return (vx, vy, vz)

def time_evolution(a, b, delta):      #a, b are the limits to simulate and delta is the parameter to select a much shorter time-step
    (c_vec, u, vT, v_ell, Ac, Bc) = init_velocity()  #vell is the modulus of v_ell to normalize the final solution
    #c_vec is the particle velocity in fluid's frame, vT is the background thermal speed and U is the fluid velocity
    (c0, Ainv) = c_quantities(c_vec) #initial speed c0 and inverse matrix
    vec = np.array([0.0, 0.0, c0]) #velocity in the fluid's frame, we rotate to z-component simply by v_z = c0
    (tau_t, tau_parallel, tau_perp) = time_scales(G, c0, vT, Ac, Bc)    #Time scales computation
    tau_ch = min(tau_t, tau_parallel, tau_perp)                         #we choose the minimum time scale as characteristic time
    dt = delta*tau_ch                    #we select dt much shorter than tau_ch
    t0 = a*tau_ch                     #initial time for the simulation
    tf = b*tau_ch                     #final time for the simulation
    Ntime = int((tf - t0) / dt)         #iteration number for the time grid
    sol = np.zeros((Ntime+1, 3))        #we save the velocity after solving the problem
    #t = np.array([])                    #here we save the time grid
    sol[0] =vec                         #we save the initial velocity
    #t = np.append(t, t0)                #we save the intial time
    #from here we are in the fluid's frame of reference
    for i in range(Ntime):
        v = linalg.norm(vec)                            #norm of the particle velocity in fluid's frame
        x = v/vT                                        #velocity ratio this is sqrt(x) = v/vT
        vec = velocity_changes(vec, dt, x, v, vT, Ac, Bc, G, v_parallel, v2_parallel, v2_perp) #it returns the new velocity
        ######### SAVING THE SOLUTION AND THE TIME GRID
        sol[i + 1] = vec                          #we save the new velocity in each time step in fluid's frame
        #t = np.append(t, t0 + (i+1)*dt)         #time grid
    sol = matrix_resul(Ainv, sol)               #rotation to the lab frame
    sol = translation(sol, u)                   #fluid velocity addition
    sol = sol/linalg.norm(v_ell)                #normalize the total solution
    return  sol

def average_values(M):
    #With function we compute the average velocity of all the ensemble
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

def sherlock_func(N):           #N is the number of particles in the beam
    delta = 0.01                #parameter to multiply the characteristic time and to select a much shorter time-step
    a = 0.0                     #lower time limit
    b = 5.0                     #upper time limit
    Ntime = int((b-a)/delta)    #number of  points to simulate the time grid
    t = np.arange(a, b + 0.01, delta)
    vx_matrix = np.zeros((N, Ntime+1))
    vy_matrix = np.zeros((N, Ntime+1))
    vz_matrix = np.zeros((N, Ntime+1))
    for i in range(N):          #We run a loop for a beam of N particles with the same initial conditions
        sol = time_evolution(a, b, delta)
        (vx, vy, vz) = components_extraction(sol)
        vx_matrix[i] = vx
        vy_matrix[i] = vy
        vz_matrix[i] = vz
    vx_mean = average_values(vx_matrix)         #average vx
    vy_mean = average_values(vy_matrix)         #average vy
    vz_mean = average_values(vz_matrix)         #average vz
    v = norm(vx_mean, vy_mean, vz_mean)         #norm of the avergae speed
    return (t, vx_mean,vy_mean,vz_mean, v)