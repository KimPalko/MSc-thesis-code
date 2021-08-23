from pendulum_env import PendulumEnv
from half_cheetah_v3_custom import HalfCheetahEnv
from humanoid_v3_custom import HumanoidEnv
from custom_ppo import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from definitions import PROJECT_ROOT
import numpy as np
import os


"""
Script for testing trained models. Allows one to choose a trained model (saved set of policy weights) and observe
the performance and behavior of an agent.
"""


def main():
    """
    Parse script arguments and load a policy to test.
    """
    exp_dir = 'pendulum/pendulum_baseline_minimal/'
    #exp_dir = 'humanoid/humanoid_10_1_a/'
    #exp_dir = 'cheetah/cheetah_a2/'
    #exp_dir = ''
    exp_name = 'pendulum_5'
    headless = False
    n_episodes = 50
    runs = next(os.walk(f'{PROJECT_ROOT}/results/{exp_dir}'))[1]
    curriculum = runs[0].split('_')[-2]
    if 'reward' in runs[0]:
        curriculum = 'reward_shaping'

    model_count = len([name for name in next(os.walk(f'{PROJECT_ROOT}/results/{exp_dir}/{runs[0]}'))[2]
                       if 'model' in name])
    model_num = ''
    if model_num != '':
        model_num = f'_{model_num}'
    elif model_count > 2:
        model_num = f'_{model_count-1}'

    model = PPO.load(f'{PROJECT_ROOT}/results/{exp_dir}{exp_name}_{curriculum}_curriculum/model{model_num}',
                     device='cpu')
    model.policy.set_std(0.01)

    if 'pendulum' in exp_dir:
        with open(f'{PROJECT_ROOT}/results/{exp_dir}{exp_name}_{curriculum}_curriculum/description.txt') as desc_file:
            lines = desc_file.readlines()
            if any([line for line in lines if 'Pendulum' in line]):
                line = [l for l in lines if 'action_weight' in l]
                action_weight = float(line[0].split('=')[1].strip())
                print(f'Action weight: {action_weight}')
        env = make_vec_env('CustomPendulum-v0', env_kwargs={'action_weight': action_weight, 'low_only': False,
                                                            'curriculum_coeff': 0})
    elif 'cheetah' in exp_dir:
        env = make_vec_env('CustomHalfCheetah-v3', env_kwargs={'assist_coeff': 0.0, 'assist_setpoint': 0})
    elif 'humanoid' in exp_dir:
        with open(f'{PROJECT_ROOT}/results/{exp_dir}{exp_name}_{curriculum}_curriculum/description.txt') as desc_file:
            lines = desc_file.readlines()
            line = [l for l in lines if 'control_cost_weight' in l]
            if len(line) == 0:
                ctrl_cost_weight = 0.1
            else:
                ctrl_cost_weight = float(line[0].split('=')[1].strip())
            print(f'Control cost weight: {ctrl_cost_weight}')
        env = make_vec_env('CustomHumanoid-v3', env_kwargs={'assist_coeff': 0.0, 'terminate_when_unhealthy': False,
                                                            'ctrl_cost_weight': ctrl_cost_weight})
    else:
        raise NotImplementedError('Unknown environment')
    envs = [name for name in next(os.walk(f'{PROJECT_ROOT}/results/{exp_dir}{exp_name}_{curriculum}_curriculum'))[2]
            if 'env' in name]
    if len(envs) > 1:
        env = VecNormalize.load(f'{PROJECT_ROOT}/results/{exp_dir}{exp_name}_{curriculum}_curriculum/env{model_num}',
                                env)
    else:
        env = VecNormalize.load(f'{PROJECT_ROOT}/results/{exp_dir}{exp_name}_{curriculum}_curriculum/env', env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()
    episode_count = 0
    ep_return = []
    rew_sum = 0
    while episode_count < n_episodes:
        action, states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        rewards = env.get_original_reward()
        rew_sum += rewards
        if not headless:
            env.render()
        if any(dones):
            episode_count += 1
            print(f'Episode return: {rew_sum}')
            ep_return.append(rew_sum)
            rew_sum = 0

    print(f'Average return: {np.average(ep_return)}')
    env.close()


if __name__ == '__main__':
    main()
