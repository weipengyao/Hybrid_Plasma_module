import module2 as md2
import matplotlib.pyplot as plt
import os

class Hybrid_Plasma:
    pass
    def __init__(self, N, delta):
        self.N = N #number of particles in the beam
        self.delta = delta #time-step
    def run_program(self):
        (t, v) = md2.sherlock_func(self.N, self.delta)
        file = open("speed_solution_{}.csv".format('data'), "w")
        for i in range(len(v)):
            file.write("{} {}".format(t[i],v[i]) + os.linesep)
        file.close()
        return (t, v)

    def run_programT(self):
        (t, Temp) = md2.sherlock_funcT(self.N, self.delta)
        file = open("temperature_solution_{}.csv".format('data'), "w")
        for i in range(len(Temp)):
            file.write("{} {}".format(t[i], Temp[i]) + os.linesep)
        file.close()
        return (t, Temp)

    def plot_results(self):
        (t, v) = self.run_program()
        plt.plot(t, v, label="v")
        plt.xlabel('t/tau')
        plt.ylabel('v/v0')
        plt.legend()
        plt.show()

    def plot_resultsT(self):
        (t, T) = self.run_programT()
        plt.plot(t, T, label="dt = 0.01")
        plt.plot([0.0,8.0],[0.1,0.1], '-.')
        plt.title("$T_i(t=0) = 10T_e$ ")
        plt.xlabel('$t/\\tau_\epsilon$')
        plt.ylabel('$T_i(t)/T_i(t=0)$')
        plt.legend()
        plt.show()

    def __del__(self):
        pass

plasma = Hybrid_Plasma(5000, 0.01)
plasma.plot_results()
#plasma.plot_resultsT()
