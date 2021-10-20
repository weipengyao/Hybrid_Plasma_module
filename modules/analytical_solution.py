## analytical_solution.py

# import the libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function of dv/dt
def model(v, t, tau):
    # tau = tau_t[0] * delta
    dvdt = - v / tau
    return dvdt

# initial condition (v0)
v0 = 1.0

# time 

a = 0.0  # time start ?
b = 8.0  # time end ?
delta = 0.01
dt = delta * 1.0e-17
# dt = delta * tau_ch
tt = np.arange(a, b + delta, delta)

# solving the given ODE
v_analytical = odeint(model, v0, tt, args=(1,))

plt.semilogy(tt, v_analytical, '-r')
plt.show()