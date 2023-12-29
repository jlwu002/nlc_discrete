import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class PathTrackingEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.speed = 2.0
        self.length = 1.0
        self.radius = 10.0
        self.dt = 0.05
        self.max_force = 0.84
        self.d_e_max = 5.0
        self.theta_e_max = math.pi

        high = np.array(
            [
                self.d_e_max,
                self.theta_e_max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        d_e, theta_e = self.state
        u = min(max(action[0], -self.max_force), self.max_force)

        costheta_e = math.cos(theta_e)
        sintheta_e = math.sin(theta_e)

        d_e_dot = self.speed * sintheta_e
        coef = self.radius / self.speed
        theta_e_dot = (u * self.speed)/self.length - costheta_e/(coef - sintheta_e)

        d_e_next = d_e + d_e_dot * self.dt
        theta_e_next = theta_e + theta_e_dot * self.dt

        d_e_next = min(max(d_e_next, -self.d_e_max), self.d_e_max)
        theta_e_next = min(max(theta_e_next, -self.theta_e_max), self.theta_e_max)

        self.state = (d_e_next, theta_e_next)
        done = False
        reward = -np.sqrt(d_e**2 + theta_e**2)/10.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        high = np.array([
                self.d_e_max,
                self.theta_e_max,
            ])
        self.state = self.np_random.uniform(low=-high, high=high)
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
                