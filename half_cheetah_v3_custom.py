import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Modified version of the OpenAI Gym Half-cheetah environment. This version introduces a P-controlled stabilizer
    for balance assistance.
    """
    def __init__(self,
                 xml_file='half_cheetah.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 assist_coeff=0.0,
                 curriculum_rate=0,
                 assist_setpoint=0):
        utils.EzPickle.__init__(**locals())

        self.assist_coeff = assist_coeff
        self.prev_angle = 0.0
        self.curriculum_rate = curriculum_rate
        self.assist_setpoint = assist_setpoint

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def set_assist_coeff(self, new_coeff):
        self.assist_coeff = new_coeff

    def balance_assist(self):
        """
        Compute and apply a balance assistance torque to the half-cheetah to assist with locomotion.

        :return: none
        """
        #self.sim.data.xfrc_applied[1][2] = 137
        sin = self.sim.data.body_xquat[1][2]
        cos = self.sim.data.body_xmat[1][0]

        diff = self.assist_setpoint - np.arctan2(sin, cos)
        if np.abs(diff - self.prev_angle) > 0.9 * np.pi:
            if np.sign(diff) == -1:
                diff += np.pi
            else:
                diff -= np.pi

        p_coeff = 30
        self.sim.data.xfrc_applied[1][4] = self.assist_coeff * p_coeff * diff

        """
        self.prev_angle = diff
        if diff >= np.pi/2:
            p_coeff = -30
            self.sim.data.xfrc_applied[1][4] = self.assist_coeff * p_coeff * diff
        elif diff <= -np.pi/2:
            p_coeff = 30
            self.sim.data.xfrc_applied[1][4] = self.assist_coeff * p_coeff * diff
        else:
            self.sim.data.xfrc_applied[1][4] = 0
        """

    def update_curriculum(self):
        """
        Update the strength of the stabilizer according to a curriculum

        :return: none
        """
        self.assist_coeff = max(self.assist_coeff - self.curriculum_rate, 0)

    def step(self, action):
        if self.assist_coeff != 0.0:
            self.balance_assist()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        self.update_curriculum()
        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


register(
    id='CustomHalfCheetah-v3',
    entry_point="%s:HalfCheetahEnv" % __name__,
    max_episode_steps=500,
)
