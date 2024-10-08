import numpy as np
import math

from draw import draw_trailer
from trailer_model import TrailerModel

class MPPIController:
    init_state = np.array([0, 0, math.radians(0), math.radians(0), math.radians(0)])      # x, y, yaw, yaw_t, steer
    goal_state = np.array([100, 100, math.radians(0), math.radians(0), math.radians(0)])

    trailer = TrailerModel(init_state.copy())  # current trailer

    n_samples = 100
    horizon = 25

    cov = [3, 1]
    controls = np.zeros((horizon, 2))
    costs = np.zeros(n_samples)

    lamb = 10
    nu = 500
    R = np.diag([1, 5])

    def __init__(self, obstacles):
        self.obstacles = obstacles

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
        distance_cost = 150 * np.linalg.norm(trailer.state[0:2] - self.goal_state[0:2]) ** 2
        if np.linalg.norm(trailer.state[0:2] - self.goal_state[0:2]) > 6:
            goal_state_weight = 0
        else:
            goal_state_weight = 500
        goal_state_cost = goal_state_weight * (abs(trailer.state[2] - self.goal_state[2]) ** 2 + abs(trailer.state[3] - self.goal_state[3]) ** 2)

        return distance_cost + goal_state_cost + obstacle_cost

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

        if min_dist_tractor < self.obstacles[min_dist_idx][2] + math.sqrt((trailer.RF + trailer.RB) ** 2 + (trailer.W / 2) ** 2):
            hit = 1
        if min_dist_trailer < self.obstacles[min_dist_idx][2] + math.sqrt((trailer.RTF + trailer.RTB) ** 2 + (trailer.W / 2) ** 2):
            hit = 1

        return 100 * math.exp(-min_dist/5) + 1e7 * hit

    def total_entropy(self, v, w, costs):
        exponents = np.exp(-1/self.lamb * costs)
        return exponents @ v / sum(exponents), exponents @ w / sum(exponents)

    def finished(self):
        if np.linalg.norm(self.goal_state[0:2] - self.trailer.state[0:2]) < 0.5 and  self.goal_state[2] - self.trailer.state[2] < np.deg2rad(3.0) :
            return True
        else:
            return False