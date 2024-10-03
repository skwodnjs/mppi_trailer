import math
import matplotlib.pyplot as plt
import numpy as np

class TrailerModel:
    MOVE_STEP = 0.2  # [m] path interpolate resolution
    TIME_STEP = 0.05  # [sec]

    W = 3.0  # [m] width of vehicle
    WB = 3.5  # [m] wheelbase: rear to front steer
    WD = 0.7 * W  # [m] distance between left-right wheels
    RF = 4.5  # [m] distance from rear to vehicle front end of vehicle
    RB = 1.0  # [m] distance from rear to vehicle back end of vehicle

    RTR = 8.0  # [m] rear to trailer wheel
    RTF = 1.0  # [m] distance from rear to vehicle front end of trailer
    RTB = 9.0  # [m] distance from rear to vehicle back end of trailer
    TR  = 0.5  # [m] tyre radius
    TW = 1.0  # [m] tyre width

    MAX_STEER = 0.6     # [rad] maximum steering angle

    MAX_VEL = 8        # [m/s] maximum velocity
    MAX_ANG_SPEED = 90  # [rad/s] maximum angular speed

    def __init__(self, state):
        self.state = state  # x, y, yaw, yaw_t, steer

    def trailer_motion_model(self, v, w):
        # x, y, yaw, yaw_t, steer
        if v > self.MAX_VEL:
            v = self.MAX_VEL

        if w > self.MAX_ANG_SPEED:
            w = self.MAX_ANG_SPEED
        elif w < -self.MAX_ANG_SPEED:
            w = -self.MAX_ANG_SPEED

        d = v * self.TIME_STEP
        steer = self.state[4] + w * self.TIME_STEP

        if steer > self.MAX_STEER:
            steer = self.MAX_STEER
        elif steer < -self.MAX_STEER:
            steer = -self.MAX_STEER

        self.state[0] += d * math.cos(self.state[2])
        self.state[1] += d * math.sin(self.state[2])
        self.state[2] += d / self.RTR * math.tan(self.state[4])
        self.state[3] += d / self.WB * math.sin(self.state[2] - self.state[3])
        self.state[4] = steer

    def draw_trailer(self, color='black', alpha=0.5):
        (x, y, yaw, yawt, steer) = self.state
        car = np.array([[-self.RB, -self.RB, self.RF, self.RF, -self.RB],
                        [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])

        trail = np.array([[-self.RTB, -self.RTB, self.RTF, self.RTF, -self.RTB],
                          [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])

        wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
                          [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])

        rlWheel = wheel.copy()
        rrWheel = wheel.copy()
        frWheel = wheel.copy()
        flWheel = wheel.copy()
        rltWheel = wheel.copy()
        rrtWheel = wheel.copy()

        Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])

        Rot2 = np.array([[math.cos(steer), -math.sin(steer)],
                         [math.sin(steer), math.cos(steer)]])

        Rot3 = np.array([[math.cos(yawt), -math.sin(yawt)],
                         [math.sin(yawt), math.cos(yawt)]])

        frWheel = np.dot(Rot2, frWheel)
        flWheel = np.dot(Rot2, flWheel)

        frWheel += np.array([[self.WB], [-self.WD / 2]])
        flWheel += np.array([[self.WB], [self.WD / 2]])
        rrWheel[1, :] -= self.WD / 2
        rlWheel[1, :] += self.WD / 2

        frWheel = np.dot(Rot1, frWheel)
        flWheel = np.dot(Rot1, flWheel)

        rrWheel = np.dot(Rot1, rrWheel)
        rlWheel = np.dot(Rot1, rlWheel)
        car = np.dot(Rot1, car)

        rltWheel += np.array([[-self.RTR], [self.WD / 2]])
        rrtWheel += np.array([[-self.RTR], [-self.WD / 2]])

        rltWheel = np.dot(Rot3, rltWheel)
        rrtWheel = np.dot(Rot3, rrtWheel)
        trail = np.dot(Rot3, trail)

        frWheel += np.array([[x], [y]])
        flWheel += np.array([[x], [y]])
        rrWheel += np.array([[x], [y]])
        rlWheel += np.array([[x], [y]])
        rrtWheel += np.array([[x], [y]])
        rltWheel += np.array([[x], [y]])
        car += np.array([[x], [y]])
        trail += np.array([[x], [y]])

        plt.plot(car[0, :], car[1, :],   color, alpha=alpha, linewidth=1)
        plt.plot(trail[0, :], trail[1, :], color, alpha=alpha, linewidth=1)
        plt.plot(frWheel[0, :], frWheel[1, :], color, alpha=alpha, linewidth=1)
        plt.plot(rrWheel[0, :], rrWheel[1, :], color, alpha=alpha, linewidth=1)
        plt.plot(flWheel[0, :], flWheel[1, :], color, alpha=alpha, linewidth=1)
        plt.plot(rlWheel[0, :], rlWheel[1, :], color, alpha=alpha, linewidth=1)
        plt.plot(rrtWheel[0, :], rrtWheel[1, :], color, alpha=alpha, linewidth=1)
        plt.plot(rltWheel[0, :], rltWheel[1, :], color, alpha=alpha, linewidth=1)

if __name__ == '__main__':
    x, y, yaw, yawt, steer = 0, 0, math.radians(0), math.radians(0), math.radians(0)
    trailer = TrailerModel([x, y, yaw, yawt, steer])
    plt.figure()
    trailer.draw_trailer(color='red', alpha=1)

    v = 8  # [m/s]
    w = 0 # [rad/s]

    for i in range(100):
        if i < 40:
            w = 0.5
        else:
            w = -0.5
        trailer.trailer_motion_model(v, w)
        trailer.draw_trailer(alpha=0.1)

    trailer.draw_trailer(color='blue', alpha=1)

    plt.xlim(-10, 100)
    plt.ylim(-10, 100)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()