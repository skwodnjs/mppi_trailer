import numpy as np
import matplotlib.pyplot as plt
import math

from trailer_model import TrailerModel

n_samples = 100
horizon = 25
lamb = 10
nu = 500
R = np.diag([1, 5])
cov = [1, 0.4]
dt = 0.1

init_state = [0, 0, 0, 0, 0]  # x, y, phi, v, steer
goal_state = [6, 6, 0]

# Define environment - obstacles
n_obstacles = 40
obstacles = np.hstack([np.random.rand(n_obstacles, 1) * 89.5 + 7.5, np.random.rand(n_obstacles, 1) * 92.5 + 4.5, 3 * np.ones((n_obstacles, 1))])

# Plot obstacles and trailer
a = plt.axes(xlim=(-10, 100), ylim=(-10, 100))
a.set_aspect('equal')

for i in range(n_obstacles):
    a.add_patch(plt.Circle(obstacles[i][0:2], obstacles[i][2], fc='b'))

plt.axhline(1.5, 0, 100, color='lightgray', linestyle='--', linewidth=1)
plt.axvline(4.5, 0, 100, color='lightgray', linestyle='--', linewidth=1)
x, y, yaw, yawt, steer = 0, 0, math.radians(0), math.radians(0), math.radians(0)
trailer = TrailerModel([x, y, yaw, yawt, steer])
trailer.draw_trailer()

plt.show()
