import module2 as md2
import matplotlib.pyplot as plt
import numpy as np

(t, v, solc,E) = md2.time_evolution()
(vx, vy, vz) = md2.components_extraction(solc) #we extract the z-component to see the slowing motion

plt.plot(t, v, label = "v")
#plt.plot(t, vy, label = "vy")
#plt.plot(t, vx, label = "vx")
plt.xlabel('t/tau')
plt.ylabel('vz/v0')
plt.legend()
plt.show()
