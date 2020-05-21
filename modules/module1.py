import module2 as md2
import matplotlib.pyplot as plt
import numpy as np
import os

'''
class Hybrid_Plasma:
    pass
    def __init__(self, N, delta):
        self.N = N #number of particles in the beam
        self.delta = delta #time-step
    def run_program(self):
        (t, vx, vy, vz, v) = md2.sherlock_func(self.N, self.delta)
        return (t, vx, vy, vz, v)

    def run_programT(self):
        (t, Temp) = md2.sherlock_funcT(self.N)
        return (t, Temp)

    def plot_results(self):
        (t, vx, vy, vz, v) = self.run_program()
        #coef = np.polyfit(t, v, 1)
        #print(coef)
        #poly1d_fn = np.poly1d(coef)
        #plt.plot(t, v, '--', label="v")
        #plt.plot(t,np.exp(-t), label = "theory")
        plt.plot(t,  np.exp(-t) - v)
        plt.title("N = 10, ")
        #plt.plot(t, vz, '-.', label="vz")
        #plt.plot(t, vy, '.', label="vy")
        #plt.plot(t, vx, '--',label="vx")
        #plt.plot(t, poly1d_fn(t), label = 'linear fit')
        plt.xlabel('t/tau')
        plt.ylabel('v/v0')
        plt.legend()
        plt.show()

    def plot_resultsT(self):
        (t, T) = self.run_programT()
        plt.plot(t, T, label="dt = 0.01")
        plt.plot([0.0,8.0],[0.1,0.1], '-.')
        plt.xlabel('t/te')
        plt.ylabel('T/T0')
        plt.legend()
        plt.show()

    def __del__(self):
        pass

plasma = Hybrid_Plasma(10, 0.01)
plasma.plot_results()
#plasma.plot_resultsT()
'''
dt = 0.01

plt.rcParams.update({'font.size': 20})
#(t, vx0, vy0, vz0, v0) = md2.sherlock_func(100, dt)
#(t, vx1, vy1, vz1, v1) = md2.sherlock_func(500, dt)
#(t, vx2, vy2, vz2, v2) = md2.sherlock_func(1000, dt)
(t, vx3, vy3, vz3, v3) = md2.sherlock_func(500, dt)
file = open("number of solutions for N={}.csv".format('pinto'), "w")
for i in range(len(vz3)):
    file.write("{} {}".format(t[i],v3[i]) + os.linesep)
file.close()

#(t0, T0) = md2.sherlock_funcT(1000, dt)
#(t1, T1) = md2.sherlock_funcT(1000, 0.1)
#(t2, T2) = md2.sherlock_funcT(1000, 0.01)

#coef = np.polyfit(t, v3, 1)
#print(coef)
#poly1d_fn = np.poly1d(coef)
#plt.plot(t, poly1d_fn(t), color = 'blue', linestyle = ':', label = 'linear fit')

#plt.plot(t,  np.exp(-t), color = 'blue',linestyle = '-', label = '$Theory$')
#plt.plot(t,  v0, color = 'blue', linestyle = ':', label = "$N = 100$")
#plt.plot(t,  v1, color = 'red', linestyle = '--', label = "$N = 500$")
#plt.plot(t,  v2, color = 'green', linestyle = '-.', label = "$N = 1000$")
plt.title("$\Delta t = 0.1\\tau_{ch}$")
plt.plot(t,  v3, color = 'black', linestyle = '-', label = "$N = 1000$")

plt.xlabel('t/$\\tau_{ch}$')
plt.ylabel('$v/v_0$')
plt.legend()
#plt.title("$N=1000, T_i(t=0) = 10T_e$")

#plt.plot(t,  abs(np.exp(-t) - v0), color = 'blue', linestyle = ':', label = "$N = 100$")
#plt.plot(t,  abs(np.exp(-t) - v1), color = 'red', linestyle = '-.', label = "$N = 500$")
#plt.plot(t,  abs(np.exp(-t) - v2), color = 'black', linestyle = '--', label = "$N = 1000$")


#plt.plot([0.0, 8.0], [0.1, 0.1], color = 'green',linestyle = '-.', label = 'Fluid')
#plt.plot(t2,  (9.0*np.exp(-2.0*t2/3.0) + 1.0)/10.0, color = 'blue',linestyle = '-', label = 'Theory')
#plt.plot(t0,  T0, color = 'magenta', linestyle = ':', label = "$\Delta$t = 0.5$\\tau_\epsilon$")
#plt.plot(t1,  T1, color = 'red', linestyle = '--', label = "$\Delta$t = 0.1$\\tau_\epsilon$")
#plt.plot(t2,  T2, color = 'black', linestyle = '-.', label = "$\Delta$t = 0.01$\\tau_\epsilon$")

#plt.plot(t0,  abs((9.0*np.exp(-2.0*t0/3.0) + 1.0)/10.0 - T0), color = 'blue', linestyle = ':', label = "$\Delta$t = 0.5$\\tau_\epsilon$")
#plt.plot(t1,  abs((9.0*np.exp(-2.0*t1/3.0) + 1.0)/10.0 - T1), color = 'red', linestyle = '-.', label = "$\Delta$t = 0.1$\\tau_\epsilon$")
#plt.plot(t2,  abs((9.0*np.exp(-2.0*t2/3.0) + 1.0)/10.0 - T2), color = 'black', linestyle = '--', label = "$\Delta$t = 0.01$\\tau_\epsilon$")

#print("error 1 = ", np.mean(abs(np.exp(-t) - v0)))
#print("error 2 = ", np.mean(abs(np.exp(-t) - v1)))
#print("error 3 = ", np.mean(abs(np.exp(-t) - v2)))
#print("error 1 = ", np.mean(abs((9.0*np.exp(-2.0*t0/3.0) + 1.0)/10.0 - T0)))
#print("error 2 = ", np.mean(abs((9.0*np.exp(-2.0*t1/3.0) + 1.0)/10.0 - T1)))
#print("error 3 = ", np.mean(abs((9.0*np.exp(-2.0*t2/3.0) + 1.0)/10.0 - T2)))
plt.show()