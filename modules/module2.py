import numpy as np
from numpy import random
from numpy import linalg
from scipy import special as sp

####################################################################################################
########################               MAIN FUNCTIONS               ################################
####################################################################################################

def parameters():
    Z = 10
    me = 9.10938291e-31 #electron mass
    mp = 1.672621898e-27 #proton mass
    mu = me/mp #electron-proton mass ratio
    nb = 1e28 # ion particle density
    ne = nb*Z #electron particle density from quasineutrality condition
    q_elec = 1.602176565e-19 #electron charge
    qt = -q_elec #test particle charge, we consider an electron
    qb = Z*qt #background ion-particle charges
    lnCol = 17.0 #Coulomb logarithm experimental value for laboratory plasma
    e_0 = 8.8541878176204e-12
    eV = 1.60217733e-19
    kTe = 50.0 #electron temperature times Boltzmann constant
    kTi = kTe/1.0

    gamma_tb = ((ne * qt**2.0 * qb**2.0)/(2.0 * np.pi * e_0**2.0 * me**2.0))*lnCol
    vT = np.sqrt(2.0*kTe*eV/me)
    # Ac and Bc are the factors to dimensionless the coefficients
    Ac = gamma_tb * (1 + mu) / vT ** 2.0  #Ac has acceleration dimensions
    Bc = gamma_tb / vT #Bc has velocity square over second, dimensions
    return (vT, gamma_tb, Ac, Bc)

#### SPECIAL FUNCTION DEFINITION ########
def G(x):
    derf = (2.0 / np.sqrt(np.pi))*np.exp(-x**2.0) #error function derivative
    return (sp.erf(x) - x*derf)/(2.0*x**2.0)

#### DIMENSIONLESS COEFFICIENTS DEFINITION #####
def v_parallel(G, x):               #frictional coeff
    return -1.0*G(x)
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
    A = [[np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)], [-np.sin(ph), np.cos(ph), 0.0], [np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)]]
    return A
def inv_rot_mat(th, ph):   #inverse rotation matrix
    A = [[np.cos(th)*np.cos(ph), -np.sin(ph), np.cos(ph)*np.sin(th)], [np.cos(th)*np.sin(ph), np.cos(ph), np.sin(th)*np.sin(ph)], [-np.sin(th), 0.0, np.cos(th)]]
    return A

### c computation ###
def c_quantities(c_vec):
    #this func computes the C quantities to get the inverse rot matrix
    c = linalg.norm(c_vec)  #norm
    c_perp = np.sqrt(c_vec[0] ** 2.0 + c_vec[1] ** 2.0) #norm in the perpendicular plane
    th = np.arccos(c_vec[2] / c)  #theta angle for the inverse matrix
    ph = np.arccos(c_vec[0] / c_perp) #phi angle for the inverse matrix
    Ainv = inv_rot_mat(th, ph) #inv rot matrix
    return (c, Ainv) #it returns the norm and the inverse matrix to come back to the lab frame

#### SCALE TIMES ####
def time_scales(G, v, vT, Ac, Bc):
    #time scales computation
    #Ac and Bc are the dimension quantities
    x = v/vT
    tau_t = v / (Ac*abs(v_parallel(G,x)))
    tau_parallel = v**2.0 / (Bc*v2_parallel(G,x))
    tau_perp = v**2.0 / (Bc*v2_perp(G,x))
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
    u = random.uniform(-1.0,1.0, 3)  #random fluid velocity in lab frame
    v_ell = random.uniform(-2.0,2.0, 3)  #random particle velocity in lab frame
    c_vec = v_ell - u #particle velocity in fluid's frame
    return  (c_vec*vT, u, vT, Ac, Bc)

def velocity_changes(dt, x, Ac, Bc, flim, v_parallel, v2_parallel, v2_perp):

    ############## DETERMINISTIC CHANGE EVOLUTION ##############
    dvz_det = dt * Ac * v_parallel(G, x)
    ############# STANDARD DEVIATION COMPUTATION  ##############
    sigma_parallel = np.sqrt(dt * Bc * v2_parallel(G, x))
    sigma_perp = np.sqrt(dt * Bc * v2_perp(G, x))
    ########## COMPUTATION OF THE STOCHASTIC CHANGES ###########
    dvz_sto = N(sigma_parallel)     #z-component stochastic change
    Np = N(sigma_perp)              #random velocity modulus in diffusion process
    angle = theta()                 #random scattering angle in diffusion process

    dvx = Np * np.cos(angle)
    dvy = Np * np.sin(angle)
    dvz = flim*dvz_det + dvz_sto         #total change in z-component
    return (dvx, dvy, dvz)

def limiting_factor(dt, G, v, vT, Ac, Bc):
    (tau_t, tau_parallel, tau_perp) = time_scales(G, v, vT, Ac, Bc)
    return min(1.0, tau_t / (2.0 * dt), tau_parallel / (2.0 * dt), tau_perp / (2.0 * dt))

def components_extraction(solc):
    #this function extact the velocity components from the total solution
    vx = np.array([])
    vy = np.array([])
    vz = np.array([])
    for i in range(len(solc)):
        vx = np.append(vx, abs(solc[i, 0]))
        vy = np.append(vy, abs(solc[i, 1]))
        vz = np.append(vz, abs(solc[i, 2]))
    return (vx, vy, vz)
####################################################################################################
####################################################################################################
####################################################################################################

def time_evolution():
    (c_vec, u, vT, Ac, Bc) = init_velocity()
    #c_vec is the particle velocity in fluid's frame, vT is the background thermal speed and U is the fluid velocity
    (c0, Ainv) = c_quantities(c_vec) #initial speed c0 and inverse matrix
    vec = np.array([0.0, 0.0, c0]) #velocity in fluid's frame, we rotate to z-component simply by v_z = c0

    (tau_t, tau_parallel, tau_perp) = time_scales(G, c0, vT, Ac, Bc) #Time scales computation
    tau_ch = min(tau_t, tau_parallel, tau_perp) #we choose the minimum time scale

    dt = 0.01*tau_ch  # we select dt much shorter than tau_ch
    t0 = 0.0*tau_ch
    tf = 10.0*tau_ch
    Ntime = int((tf - t0) / dt)  #iteration number
    sol = np.zeros((Ntime+1, 3)) #we save the velocity after solving the problem
    t = np.array([])   #here we save the time grid

    #from here we are in the fluid's frame of reference
    for i in range(Ntime+1):

        v = linalg.norm(vec)  #norm of particle velocity in fluid's frame
        x = v/vT  #velocity ratio

        flim = limiting_factor(dt, G, v, vT, Ac, Bc)
        (dvx, dvy, dvz) = velocity_changes(dt, x, Ac, Bc, flim, v_parallel, v2_parallel, v2_perp)

        ########## ADDING THE CHANGES IN VELOCITY
        vec[0] = vec[0] + dvx
        vec[1] = vec[1] + dvy
        vec[2] = vec[2] + dvz
        ######### COMPUTATION OF THE LIMITING FACTOR

        sol[i] = vec  # we save the new velocity in each time step
        t = np.append(t, t0 + i*dt)

    sol = matrix_resul(Ainv, sol)
    sol = translation(sol, u)
    sol = sol/c0   #normalize the total solution
    aux = norma_matrix(sol)
    return (t/tau_ch, aux, sol)