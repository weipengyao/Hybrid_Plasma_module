import numpy as np
from numpy import random
from numpy import linalg
from scipy import special as sp

####################################################################################################
########################               MAIN FUNCTIONS               ################################
####################################################################################################

def parameters():
    Z = 1
    Zb = 7
    me = 9.10938291e-31                     #electron mass
    mp = 1.672621898e-27                    #proton mass
    mu = mp/me                              #electron-proton mass ratio
    nb = 7e23                               #ion particle density
    q_elec = 1.602176565e-19                #electron charge
    qt = Z*q_elec                           #test particle charge, we consider an electron
    qb = -Zb*qt                             #background ion-particle charges
    lnCol = 17.0                            #Coulomb logarithm experimental value for laboratory plasma
    e_0 = 8.8541878176204e-12               #vacuum permittivity
    eV = 1.60217733e-19                     #eV to Jules
    kT = 1                                  #electron temperature times Boltzmann constant

    gamma_tb = (qt**2.0 * qb**2.0 * nb * lnCol)/(2.0 * np.pi * e_0**2.0 * me**2.0)  #Gamma_tb factor
    vT = np.sqrt(2.0*kT*eV/me)                                                     #thermal speed
    # Ac and Bc are the factors to dimensionless the coefficients
    Ac = (gamma_tb / (vT**2.0)) * (1.0 + mu)            #Ac has acceleration dimensions
    Bc = gamma_tb / vT                              #Bc has dimensions of velocity square over second
    return (vT, gamma_tb, Ac, Bc)

#### SPECIAL FUNCTION DEFINITION ########
def G(x):
    derf = (2.0 / np.sqrt(np.pi))*np.exp(-x**2.0) #error function derivative
    return (sp.erf(x) - x*derf)/(2.0*x**2.0)

#### DIMENSIONLESS COEFFICIENTS DEFINITION #####
def v_parallel(G, x):               #frictional coeff
    return -G(x)
    #return -x / (2.0 * x ** 3.0 + (3.0 / (2.0 * np.sqrt(np.pi))) ) #Sherlock's approx

def v2_parallel(G, x):              #parallel diffusion coeff
    return G(x)/x
    #return 1.0/(2.0 * x**3.0 + (3.0 / (2.0*np.sqrt(np.pi)) )) #Sherlock's approx
def v2_perp(G, x):                  #perpendicular diffusion coeff
    return (sp.erf(x) - G(x))/x
    #return sp.erf(x)/x - 1.0 / (2.0 * x ** 3.0 + (3.0 / (2.0 * np.sqrt(np.pi)))) #Sherlock's approx

#### random functions #####
def N(sigma): #we generate a random number with a centered normal distribution with standard deviation sigma
    return random.normal(0.0, sigma)
def theta():
    return random.uniform(0.0, 2.0*np.pi)

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
    A = rot_mat(th,ph)
    Ainv = inv_rot_mat(th, ph) #inv rot matrix
    return (c, Ainv,A) #it returns the norm and the inverse matrix to come back to the lab frame

#### SCALE TIMES ####
def time_scales(G, v, vT, Ac, Bc):
    #Ac and Bc are the dimension quantities
    #If we use the approximations for special functions, you can get negative values and it's necessary to take the abs value
    x = v/vT
    tau_t = v / (Ac*abs(v_parallel(G, x)))
    tau_parallel = v**2.0 / (Bc*abs(v2_parallel(G, x)))
    tau_perp = v**2.0 / (Bc*abs(v2_perp(G, x)))
    return(tau_t, tau_parallel, tau_perp)

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
    u = vT*random.uniform(0.0,1.0, 3)  #random fluid velocity in lab frame
    v_ell = vT*random.uniform(0.0,2.0, 3)  #random particle velocity in lab frame
    #u = vT * np.array([0.0, 0.0, 0.0])
    #v_ell = vT * np.array([3e5 / (3*vT), 3e5 / (3*vT), 1.5e5 / (3*vT)])
    c_vec = v_ell - u #particle velocity in fluid's frame
    return(c_vec, u, vT, v_ell, Ac, Bc)

def velocity_changes(dt, x, Ac, Bc, flim, v_parallel, v2_parallel, v2_perp):
    ############## DETERMINISTIC EVOLUTION ##############
    dvz_det = dt * Ac * v_parallel(G, x)
    ############# STANDARD DEVIATIONS COMPUTATION  ##############
    sigma_parallel = np.sqrt(dt * Bc * abs(v2_parallel(G, x)))
    sigma_perp = np.sqrt(dt * Bc * abs(v2_perp(G, x)))
    ########## COMPUTATION OF THE STOCHASTIC CHANGES ###########
    dvz_sto = N(sigma_parallel)             #z-component stochastic change
    dV_perp = N(sigma_perp)                   #random velocity modulus in diffusion process
    angle_perp = theta()                                 #random scattering angle in diffusion process

    dvx = dV_perp * np.cos(angle_perp)                     #x-component stochastic change
    dvy = dV_perp * np.sin(angle_perp)                     #y-component stochastic change
    dvz = dvz_det + flim*(dvz_sto)                  #total change in z-component
    return np.array([dvx, dvy, dvz])                #we return a vector that contains the changes in velocity

def limiting_factor(dt, G, v, vT, Ac, Bc):
    (tau_t, tau_parallel, tau_perp) = time_scales(G, v, vT, Ac, Bc)
    return min(1.0, tau_t / (2.0 * dt), tau_parallel / (2.0 * dt), tau_perp / (2.0 * dt))

def denergy(E0, v):
    E = np.array([])
    mp = 1.672621898e-27
    for i in range(len(v)):
        E = np.append(E, abs(0.5*mp*v[i]**2.0 - E0))
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

def time_evolution():
    (c_vec, u, vT, v_ell, Ac, Bc) = init_velocity()  #vell is the modulus of v_ell to normalize the final solution
    #c_vec is the particle velocity in fluid's frame, vT is the background thermal speed and U is the fluid velocity
    (c0, Ainv, A) = c_quantities(c_vec) #initial speed c0 and inverse matrix
    vec = np.array([0.0, 0.0, c0]) #velocity in the fluid's frame, we rotate to z-component simply by v_z = c0
    (tau_t, tau_parallel, tau_perp) = time_scales(G, c0, vT, Ac, Bc)    #Time scales computation
    tau_ch = min(tau_t, tau_parallel, tau_perp)                         #we choose the minimum time scale as characteristic time
    dt = 0.01*tau_ch                #we select dt much shorter than tau_ch
    t0 = 0.0*tau_ch                 #initial time for the simulation
    tf = 4.0*tau_ch                 #final time for the simulation
    Ntime = int((tf - t0) / dt)     #iteration number for the time grid
    sol = np.zeros((Ntime+1, 3))    #we save the velocity after solving the problem
    t = np.array([])                #here we save the time grid
    sol[0] =vec                     #we save the initial velocity
    t = np.append(t, t0)            #we save the intial time
    E0 = 0.5*1.672621898e-27*c0**2.0
    #from here we are in the fluid's frame of reference
    for i in range(Ntime):
        v = linalg.norm(vec)                            #norm of the particle velocity in fluid's frame
        x = v/vT                                        #velocity ratio this is sqrt(x) = v/vT
        flim = limiting_factor(dt, G, v, vT, Ac, Bc)    #limiting factor computation
        #if flim<1:
        #    print(flim)
        dv = velocity_changes(dt, x, Ac, Bc, flim, v_parallel, v2_parallel, v2_perp)
        ########## ADDING THE CHANGES IN VELOCITY
        vec = vec + dv
        ######### SAVING THE SOLUTION AND THE TIME GRID
        sol[i+1] = vec                          #we save the new velocity in each time step in fluid's frame
        t = np.append(t, t0 + (i+1)*dt)         #time grid
    sol = matrix_resul(Ainv, sol)               #rotation to the lab frame
    sol = translation(sol, u)                   #fluid velocity addition
    E = denergy(E0, norma_matrix(sol))
    sol = sol/linalg.norm(v_ell)                #normalize the total solution
    sol_norm = norma_matrix(sol)                #norm of velocity in each time step
    return (t/tau_ch, sol_norm, sol,E)