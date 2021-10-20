import module2 as md2
import matplotlib.pyplot as plt
import os
plt.rcParams.update({'font.size': 20})

class Hybrid_Plasma: #Hybrid plasma class
    pass
    def __init__(self, N, delta):
        self.N = N #number of particles in the beam
        self.delta = delta #time-step
    def run_program(self):
        #This function runs the first test for N particles with time step dt
        (t, v) = md2.sherlock_func(self.N, self.delta)
        #The solution is saved in a CSV file
        file = open("speed_solution_{}.csv".format('data'), "w")
        for i in range(len(v)):
            file.write("{} {}".format(t[i],v[i]) + os.linesep)
        file.close()
        return (t, v)

    def run_program_test(self):
        #This function runs the first test for N particles with time step dt
        (t, v, tau_t, tau_parallel, tau_perp, flim, dt) = md2.sherlock_func_test(self.N, self.delta)

        def run_analytical_solution(delta):
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
            v0 = v[0]

            # time 
            tt = t

            # solving the given ODE
            v_analytical = odeint(model, v0, tt, args=(delta*100,))
            # print(np.shape(v_analytical))

            return v_analytical

        v_analytical = run_analytical_solution(self.delta)


        #The solution is saved in a CSV file
        file = open("speed_solution_test_{}.csv".format('data'), "w")
        for i in range(len(v)):
            file.write("{} {} {} {} {} {} {} {}".format(t[i], v[i], tau_t[i], tau_parallel[i], tau_perp[i], v_analytical[i], flim[i], dt[i]) + os.linesep)
        file.close()

        return (t, v, tau_t, tau_parallel, tau_perp, v_analytical, flim, dt)

    def run_programT(self):
        # This function runs the first test for N particles with time step dt
        (t, Temp) = md2.sherlock_funcT(self.N, self.delta)
        # The solution is saved in a CSV file
        file = open("temperature_solution_{}.csv".format('data'), "w")
        for i in range(len(Temp)):
            file.write("{} {}".format(t[i], Temp[i]) + os.linesep)
        file.close()
        return (t, Temp)
    def plot_results(self):
        # (t, v) = self.run_program()
        (t, v, tau_t, tau_parallel, tau_perp, v_analytical, flim, dt) = self.run_program_test()
        plt.plot(t, v,            color = 'black', linestyle = '-', label = "$N = 100$", linewidth = 0.5)
        plt.plot(t, v_analytical, color = 'red', linestyle = '--', label = "analytical", linewidth = 1.0)        
        plt.title("$\Delta t = 0.01\\tau_{ch}$")
        plt.xlabel('t/$\\tau_{ch}$')
        plt.xlim([0,8])
        plt.ylabel('$v/v_0$')
        # plt.ylim(0.0001,1)
        plt.legend(fancybox=False,frameon=False)
        plt.savefig('./test2_N100_dt0.001_lnCol17_Te1_ne7e23_qb1_mi27_ana_lineary.png',transparent=True,bbox_inches='tight')
        # plt.legend()
        # plt.show()
    def plot_results_test_flim(self):
        (t, v, tau_t, tau_parallel, tau_perp, v_analytical, flim, dt) = self.run_program_test()
        plt.plot(t, flim,        '-k', label='flim')
        # plt.semilogy(t, tau_parallel, '-b', label='tau_parallel')
        # plt.semilogy(t, tau_perp,     '-g', label='tau_perp')
        plt.xlabel('t/$\\tau_{ch}$')
        plt.xlim([0,8])
        plt.ylabel('$f_{lim}$')
        plt.legend(fancybox=False,frameon=False)
        plt.savefig('./test2_timescales_lnCol17_Te1_ne7e23_qb1_mi27_flim.png',transparent=True,bbox_inches='tight')

    def plot_results_test_dt(self):
        (t, v, tau_t, tau_parallel, tau_perp, v_analytical, flim, dt) = self.run_program_test()
        plt.plot(t, dt,        '--k', label='dt')
        # plt.semilogy(t, tau_parallel, '-b', label='tau_parallel')
        # plt.semilogy(t, tau_perp,     '-g', label='tau_perp')
        plt.xlabel('t/$\\tau_{ch}$')
        plt.xlim([0,8])
        plt.ylabel('dt [s]')
        plt.legend(fancybox=False,frameon=False)
        plt.savefig('./test2_timescales_lnCol17_Te1_ne7e23_qb1_mi27_dt.png',transparent=True,bbox_inches='tight')


    def plot_results_test_time(self):
        (t, v, tau_t, tau_parallel, tau_perp, v_analytical,flim,dt) = self.run_program_test()
        plt.semilogy(t, tau_t,        '-r',  label='tau_t')
        plt.semilogy(t, tau_parallel, '-b',  label='tau_parallel')
        plt.semilogy(t, tau_perp,     '-g',  label='tau_perp')
        plt.semilogy(t, dt,           '--k', label='dt')
        plt.xlabel('t/$\\tau_{ch}$')
        plt.xlim([0,8])
        plt.ylabel('$\\tau$ [s]')
        plt.legend(fancybox=False,frameon=False)
        plt.savefig('./test2_timescales_lnCol17_Te1_ne7e23_qb1_mi27.png',transparent=True,bbox_inches='tight')



    def plot_resultsT(self):
        (t, T) = self.run_programT()
        plt.plot(t, T, color = 'red', linestyle = '-', label = "$\\Delta t = 0.01$", linewidth = 1)
        plt.plot([0.0,8.0],[0.1,0.1], color = 'black', linestyle = '-.', label = "Fluid", linewidth = 1)
        plt.title("$T_i(t=0) = 10T_e$ ")
        plt.xlabel('$t/\\tau_\epsilon$')
        plt.ylabel('$T_i(t)/T_i(t=0)$')
        plt.legend()
        plt.show()
    def __del__(self):
        pass

#We create an object "Plasma" to run the first test
plasma = Hybrid_Plasma(100, 0.01)
# plasma.plot_results_test_flim()     # test for checking timescales
# plasma.plot_results_test_dt()
# plasma.plot_results_test_time()
plasma.plot_results()
#We create an second object "PlasmaT" to run the second test
#uncomment the following two lines to plot the temperature equilibration plot and comment the first test.
#plasmaT = Hybrid_Plasma(1000, 0.01)
#plasmaT.plot_resultsT()
