import numpy as np
import math
import matplotlib.pyplot as plt
from trailer_model import TrailerModel

def draw_trailer(state, color='black', alpha=0.5):
    # state: x, y, yaw, yaw_t, steer
    (x, y, yaw, yawt, steer) = state
    car = np.array([[-TrailerModel.RB, -TrailerModel.RB, TrailerModel.RF, TrailerModel.RF, -TrailerModel.RB],
                    [TrailerModel.W / 2, -TrailerModel.W / 2, -TrailerModel.W / 2, TrailerModel.W / 2, TrailerModel.W / 2]])

    trail = np.array([[-TrailerModel.RTB, -TrailerModel.RTB, TrailerModel.RTF, TrailerModel.RTF, -TrailerModel.RTB],
                      [TrailerModel.W / 2, -TrailerModel.W / 2, -TrailerModel.W / 2, TrailerModel.W / 2, TrailerModel.W / 2]])

    wheel = np.array([[-TrailerModel.TR, -TrailerModel.TR, TrailerModel.TR, TrailerModel.TR, -TrailerModel.TR],
                      [TrailerModel.TW / 4, -TrailerModel.TW / 4, -TrailerModel.TW / 4, TrailerModel.TW / 4, TrailerModel.TW / 4]])

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

    frWheel += np.array([[TrailerModel.WB], [-TrailerModel.WD / 2]])
    flWheel += np.array([[TrailerModel.WB], [TrailerModel.WD / 2]])
    rrWheel[1, :] -= TrailerModel.WD / 2
    rlWheel[1, :] += TrailerModel.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    rltWheel += np.array([[-TrailerModel.RTR], [TrailerModel.WD / 2]])
    rrtWheel += np.array([[-TrailerModel.RTR], [-TrailerModel.WD / 2]])

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

    plt.plot(car[0, :], car[1, :], color, alpha=alpha, linewidth=1)
    plt.plot(trail[0, :], trail[1, :], color, alpha=alpha, linewidth=1)
    plt.plot(frWheel[0, :], frWheel[1, :], color, alpha=alpha, linewidth=1)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color, alpha=alpha, linewidth=1)
    plt.plot(flWheel[0, :], flWheel[1, :], color, alpha=alpha, linewidth=1)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color, alpha=alpha, linewidth=1)
    plt.plot(rrtWheel[0, :], rrtWheel[1, :], color, alpha=alpha, linewidth=1)
    plt.plot(rltWheel[0, :], rltWheel[1, :], color, alpha=alpha, linewidth=1)

def draw_sample(states, color='black'):
    plt.plot(np.array([state[0] for state in states]), np.array([state[1] for state in states]), '-', color=color, alpha=0.5, linewidth=1)


if __name__ == '__main__':
    x, y, yaw, yawt, steer = 0, 0, math.radians(0), math.radians(0), math.radians(0)
    trailer = TrailerModel([x, y, yaw, yawt, steer])
    plt.figure()
    draw_trailer(trailer.state, color='red', alpha=1)

    v = 8  # [m/s]
    w = 0 # [rad/s]

    for i in range(25):
        if i < 10:
            w = 0.1
        else:
            w = -0.1
        trailer.step(v, w)
        draw_trailer(trailer.state, alpha=0.1)

    draw_trailer(trailer.state, color='blue', alpha=1)

    plt.xlim(-10, 100)
    plt.ylim(-10, 100)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()