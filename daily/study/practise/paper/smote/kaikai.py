from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = Axes3D(fig)
t = np.arange(0, 1, 0.01)
p = np.arange(1, 0, -0.01)
x, y = np.meshgrid(t, p)
z = (1-x)+2*y+1.0/3*(3.85-x+y)**2+(3.85-x+y)*(x-3.85)+(x-3.85)**2

ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')

plt.show()