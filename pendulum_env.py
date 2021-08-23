import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    """
    Modified version of the OpenAI Gym Pendulum environment. Changes introduced in this version:

        - Modified the physics computations
        - Lighter pendulum an higher maximum torque; swinging to gain momentum is not needed to get to the top
        - Option to use three different curricula: "reward shaping", "balance assistance", and "reverse curriculum"
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 low_only=False,
                 action_weight=1,
                 g=9.81,
                 curriculum_coeff=0.0,
                 curriculum_rate=0.0,
                 curriculum_type=None):
        self.max_speed = 8
        self.max_torque = 3.0
        self.dt = .1
        self.g = g
        self.m = 1.0
        self.l = 0.2
        self.viewer = None
        self.steps_done = 0
        self.max_steps = 200
        self.sample_high = np.array([np.pi, 1.0])
        self.low_only = low_only
        self.action_weight = action_weight
        self.curriculum_coeff = curriculum_coeff
        self.curriculum_rate = curriculum_rate
        self.curriculum_type = curriculum_type

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def set_max_steps(self, steps):
        self.max_steps = steps

    def set_sample_high(self, high):
        """
        Set the range for sampling the starting state.

        :param high: Numpy array of the form [angle, angular velocity]
        :return: none
        """
        self.sample_high = high

    def set_action_weight(self, weight):
        """
        Set the weight of the action cost

        :param weight: new action weight
        :return: none
        """
        self.action_weight = weight

    def set_assist_coeff(self, value):
        """
        Set the assistance coefficient value.

        :param value: new coefficient value
        :return: none
        """
        self.assist_coeff = value

    def get_info(self):
        return self.sample_high, self.action_weight

    def balance_assistance(self):
        """
        Compute the assistance torque (if using an assistance curriculum)
        :return: computed torque, clipped to the range defined by max_torque and the curriculum phase
        """
        th, thdot = self.state
        pos_error = -angle_normalize(th)
        max_assistance = self.max_torque * self.curriculum_coeff
        kp = 0.75
        kd = 1.0
        return np.clip((kp * pos_error - kd * thdot)*self.curriculum_coeff, -max_assistance, max_assistance)

    def update_curriculum(self):
        """
        Update the curriculum parameter
        :return: none
        """
        self.curriculum_coeff = max(self.curriculum_coeff - self.curriculum_rate, 0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        frict_coeff = -0.75
        max_friction = 0.5

        u = np.clip(u.item(), -self.max_torque, self.max_torque)
        #u = np.interp(u.item(), [-1, 1], [-self.max_torque, self.max_torque])
        # Increase coefficient of u to strengthen false optimum
        # costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + action_weight * (u ** 2)

        if self.curriculum_type == 'reward_shaping':
            weight_anneal_factor = self.curriculum_coeff
        else:
            weight_anneal_factor = 0.0

        action_weight = (2.0 - self.action_weight) * weight_anneal_factor + self.action_weight

        costs = angle_normalize(th)**2 + action_weight * u**2

        if self.curriculum_coeff != 0 and self.curriculum_type == 'assist':
            u = u + self.balance_assistance()
        self.last_u = u  # for rendering
        #rewards = np.exp(-(th**2)) + action_weight * np.exp(-(u**2))

        friction = np.clip(frict_coeff * thdot, -max_friction, max_friction)

        #newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u + friction) * dt
        newthdot = thdot + dt * (u + 0.5 * l * g * m * np.sin(angle_normalize(th)) + friction)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        """
        self.steps_done += 1
        if self.steps_done >= self.max_steps:
            # return self._get_obs(), -costs, True, {}
            return self._get_obs(), rewards, True, {}

        # return self._get_obs(), -costs, False, {}
        """
        if self.curriculum_type is not None and self.curriculum_type != 'stdev':
            self.update_curriculum()

        return self._get_obs(), -costs, False, {}
        #return self._get_obs(), rewards, False, {}

    def reset(self):
        if self.low_only:
            self.state = -np.array([np.pi, 0])
        else:
            if self.curriculum_type == 'reverse':
                init_state_coeff = 1 - self.curriculum_coeff
            else:
                init_state_coeff = 1
            self.state = self.np_random.uniform(low=-self.sample_high * init_state_coeff,
                                                high=self.sample_high * init_state_coeff)
        self.last_u = None
        self.steps_done = 0
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


register(
    id="CustomPendulum-v0",
    entry_point="%s:PendulumEnv" % __name__,
    max_episode_steps=200,
)
