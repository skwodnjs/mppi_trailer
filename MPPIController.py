import numpy as np
import matplotlib.pyplot as plt
import math

from draw import draw_trailer, draw_sample
from trailer_model import TrailerModel

class MPPIController:
    init_state = np.array([0, 0, math.radians(0), math.radians(0), math.radians(0)])      # x, y, yaw, yaw_t, steer
    goal_state = np.array([100, 100, math.radians(0), math.radians(0), math.radians(0)])

    trailer = TrailerModel(init_state)  # current trailer

    n_samples = 100
    horizon = 25

    cov = [1, 0.4]
    controls = np.zeros((horizon, 2))
    costs = np.zeros(n_samples)

    lamb = 10
    nu = 500
    R = np.diag([1, 5])

    def __init__(self, obstacles):
        self.obstacles = obstacles
        draw_trailer(self.init_state, color='#6666FF', alpha=1.0)
        draw_trailer(self.goal_state, color='#6666FF', alpha=1.0)

    def get_action(self):
        self.costs.fill(0)
        v = np.random.normal(loc = 0, scale = self.cov[0], size = (self.n_samples, self.horizon))
        w = np.random.normal(loc = 0, scale = self.cov[1], size = (self.n_samples, self.horizon))

        for n in range(self.n_samples):
            sample_trailer = TrailerModel(self.trailer.state.copy())
            states = [sample_trailer.state.copy()]   # horizon + 1 states (first is init point)

            for i in range(self.horizon):
                sample_trailer.step(self.controls[i][0] + v[n][i], self.controls[i][1] + w[n][i])
                states.append(sample_trailer.state.copy())
                self.costs[n] += self.get_cost(sample_trailer, self.controls[i], np.array([v[n][i], w[n][i]]))

        self.costs -= min(self.costs)

        for j in range(self.horizon):
            self.controls[j] += self.total_entropy(v[:, j], w[:, j], self.costs)
            self.controls[j][0] = max(min(self.controls[j][0], TrailerModel.MAX_VEL), 0)
            self.controls[j][1] = min(self.controls[j][1], TrailerModel.MAX_ANG_SPEED)

        self.trailer.step(self.controls[0][0], self.controls[0][1])
        draw_trailer(self.trailer.state)
        self.controls = np.vstack((self.controls[1:], np.array([0, 0])))

    def get_cost(self, trailer, control, du):
        state_cost = self.state_cost_function(trailer)
        control_cost = self.control_cost_function(control, du)

        return state_cost + control_cost

    def state_cost_function(self, trailer):
        obstacle_cost = self.obstacle_cost_function(trailer)
        heading_cost = 50 * abs(trailer.state[2] - self.goal_state[2]) ** 2
        distance_cost = 100 * np.linalg.norm(trailer.state[0:2] - self.goal_state[0:2]) ** 2

        return distance_cost + heading_cost + obstacle_cost

    def control_cost_function(self, control, du):
        return (1 - 1/self.nu) / 2 * du.T @ self.R @ du + control.T @ self.R @ du + 1/2 * control.T @ self.R @ control

    def obstacle_cost_function(self, trailer):
        dist_tractor = np.sqrt(np.sum((trailer.center_tractor - self.obstacles[:, :2]) ** 2, axis=1))
        dist_trailer = np.sqrt(np.sum((trailer.center_trailer - self.obstacles[:, :2]) ** 2, axis=1))

        hit = 0

        min_dist_tractor = np.min(dist_tractor)
        min_dist_idx = np.argmin(dist_tractor)

        min_dist_trailer = dist_trailer[min_dist_idx]
        min_dist = min_dist_tractor + min_dist_trailer

        if min_dist_tractor < obstacles[min_dist_idx][2] + math.sqrt((trailer.RF + trailer.RB) ** 2 + (trailer.W / 2) ** 2):
            hit = 1
        if min_dist_trailer < obstacles[min_dist_idx][2] + math.sqrt((trailer.RTF + trailer.RTB) ** 2 + (trailer.W / 2) ** 2):
            hit = 1

        return 230 * math.exp(-min_dist/5) + 1e6 * hit

    def total_entropy(self, v, w, costs):
        exponents = np.exp(-1/self.lamb * costs)
        return exponents @ v / sum(exponents), exponents @ w / sum(exponents)

    def finished(self):
        if np.linalg.norm(self.goal_state[0:2] - self.trailer.state[0:2]) < 0.5 and  self.goal_state[2] - self.trailer.state[2] < np.deg2rad(3.0) :
            return True
        else:
            return False

if __name__ == '__main__':
    plt.figure()

    n_obstacles = 10
    obstacles_radius = 3
    obstacles = np.hstack([np.random.rand(n_obstacles, 1) * (100 - TrailerModel.RF - TrailerModel.RTB - 2 * obstacles_radius) + TrailerModel.RF + obstacles_radius,
                           np.random.rand(n_obstacles, 1) * (100 - TrailerModel.W / 2 - 20 - 2 * obstacles_radius) + 20 + obstacles_radius,
                           obstacles_radius * np.ones((n_obstacles, 1))])

    a = plt.axes(xlim=(-10, 100), ylim=(-10, 100))
    a.set_aspect('equal')

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

    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()