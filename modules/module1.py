import module2 as md2
import matplotlib.pyplot as plt

(t, v, solc) = md2.time_evolution()
(vx, vy, vz) = md2.components_extraction(solc) #we extract the z-component to see the slowing motion

plt.plot(t, vz)
plt.xlabel('t/tau')
plt.ylabel('vz/v0')
plt.show()

