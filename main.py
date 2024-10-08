import numpy as np
import matplotlib.pyplot as plt

from MPPIController import MPPIController
from draw import draw_trailer
from trailer_model import TrailerModel

a = plt.axes(xlim=(-10, 100), ylim=(-10, 100))
a.set_aspect('equal')

# Define environment - obstacles
n_obstacles = 10
obstacles_radius = 3
obstacles = np.hstack([np.random.rand(n_obstacles, 1) * (100 - TrailerModel.RF - TrailerModel.RTB - 2 * obstacles_radius) + TrailerModel.RF + obstacles_radius,
                       np.random.rand(n_obstacles, 1) * (100 - TrailerModel.W / 2 - 20 - 2 * obstacles_radius) + 20 + obstacles_radius,
                       obstacles_radius * np.ones((n_obstacles, 1))])

for i in range(n_obstacles):
    a.add_patch(plt.Circle(obstacles[i][0:2], obstacles[i][2], fc='b'))

plt.axhline(20, 0, 100, color='lightgray', linestyle='--', linewidth=1)
plt.axvline(TrailerModel.RF, 0, 100, color='lightgray', linestyle='--', linewidth=1)
plt.axhline(100 - TrailerModel.W / 2, 0, 100, color='lightgray', linestyle='--', linewidth=1)
plt.axvline(100 - TrailerModel.RTB, 0, 100, color='lightgray', linestyle='--', linewidth=1)

controller = MPPIController(obstacles)

for i in range(300):
    controller.get_action()
    if controller.finished():
        break

draw_trailer(controller.init_state, color='#6666FF', alpha=1.0)
draw_trailer(controller.goal_state, color='#6666FF', alpha=1.0)

plt.xlim(-10, 110)
plt.ylim(-10, 110)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()