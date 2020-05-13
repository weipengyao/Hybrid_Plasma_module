import module2 as md2
import matplotlib.pyplot as plt

class Hybrid_Plasma:
    pass
    def __init__(self, N):
        self.N = N #particle number in the beam

    def run_program(self):
        (t, vx, vy, vz, v) = md2.sherlock_func(self.N)
        return (t, vx, vy, vz, v)

    def plot_results(self):
        (t, vx, vy, vz, v) = self.run_program()
        plt.plot(t, v, label="v")
        plt.xlabel('t/tau')
        plt.ylabel('v/v0')
        plt.legend()
        plt.show()

    def __del__(self):
        pass

plasma = Hybrid_Plasma(500)
plasma.plot_results()
