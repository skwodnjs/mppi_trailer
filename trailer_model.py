import math

class TrailerModel:
    MOVE_STEP = 0.2  # [m] path interpolate resolution
    TIME_STEP = 0.2  # [sec]

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
    MAX_ANG_SPEED = 0.5  # [rad/s] maximum angular speed

    def __init__(self, state):
        self.state = state  # x, y, yaw, yaw_t, steer
        self.center_tractor = (state[0] + self.WB / 2 * math.cos(state[2]), state[1] + self.WB / 2 * math.sin(state[2]))
        self.center_trailer = (state[0] - self.RTR / 2 * math.cos(state[3]), state[1] - self.RTR / 2 * math.sin(state[3]))

    def step(self, v, w):
        # x, y, yaw, yaw_t, steer
        if v > self.MAX_VEL:
            v = self.MAX_VEL
        elif v < 0:
            v = 0
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

        self.center_tractor = (self.state[0] + self.WB / 2 * math.cos(self.state[2]), self.state[1] + self.WB / 2 * math.sin(self.state[2]))
        self.center_trailer = (self.state[0] - self.RTR / 2 * math.cos(self.state[3]), self.state[1] - self.RTR / 2 * math.sin(self.state[3]))