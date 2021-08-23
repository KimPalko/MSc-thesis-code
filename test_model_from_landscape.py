from pendulum_env import PendulumEnv
from half_cheetah_v3_custom import HalfCheetahEnv
from humanoid_v3_custom import HumanoidEnv
import torch
import argparse
from custom_ppo import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from definitions import PROJECT_ROOT
import numpy as np
import os
import imageio
from sklearn.decomposition import PCA


"""
Script for picking and testing policies corresponding to points in an optimization landscape. Use contour plots
of optimization landscapes as a reference when choosing the points.
"""


def compute_new_model_params(ax1, ax2, ax1_pos, ax2_pos, opt_model):
    """
    Get the policy parameter dict corresponding to the specified point, given axes ax1, ax2.
    :param ax1: first basis vector
    :param ax2: second basis vector
    :param ax1_pos: position in the direction of ax1
    :param ax2_pos: position in the direction of ax2
    :param opt_model: optimal saved model (i.e. origin of landscape slice)
    :return: dict of policy parameters corresponding to point (ax1_pos, ax2_pos)
    """
    opt_params_dict = opt_model.get_parameters()
    opt_params_flat = np.concatenate([v.cpu().flatten() for v in opt_params_dict['policy'].values()])
    new_params_flat = opt_params_flat + ax1_pos * ax1 + ax2_pos * ax2
    new_params_dict = opt_params_dict
    dim_counter = 0
    for key, value in new_params_dict['policy'].items():
        flat_val_dim = value.flatten().shape[0]
        new_params_dict['policy'][key] = np.reshape(new_params_flat[dim_counter:dim_counter + flat_val_dim],
                                                    value.shape)
        new_params_dict['policy'][key] = torch.from_numpy(new_params_dict['policy'][key]).float()
        dim_counter += flat_val_dim

    return new_params_dict


def main():
    """
    Parse script arguments and test a policy from the specified point in an optimization landscape.
    """
    parser = argparse.ArgumentParser('Test a model with parameters picked from visualized landscape')
    parser.add_argument('--env-name', type=str, default='CustomPendulum-v0', help='Gym environment name')
    #parser.add_argument('--env-name', type=str, default='CustomHalfCheetah-v3', help='Gym environment name')
    #parser.add_argument('--env-name', type=str, default='CustomHumanoid-v3', help='Gym environment name')
    parser.add_argument('--exp-dir', type=str, default='results/pendulum/pendulum_rs', help='Name of the experiment')
    parser.add_argument('--run-name', type=str, default='pendulum_4', help='Name of the run')
    parser.add_argument('--basis-method', type=str, default='pca',
                        help='Method for computing basis vectors, "model" (default) or "pca"')
    parser.add_argument('--landscape-dir', type=str, default='plots/data/pendulum_rsc_pca_full',
                        help='Directory of the landscape data relative to project root (needed for optimization paths)')
    parser.add_argument('--param-coords', type=str, default='15.0,6.5',
                        help='Coordinates of the desired parameters as two comma-separated float values:'
                             'x units in 1st basis direction, y units in 2nd basis direction')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Toggle to run episodes without rendering')
    parser.add_argument('--stdev', type=float, default=0.0,
                        help='Standard deviation of the action noise. Set to zero for deterministic policy.')
    parser.add_argument('--record-video', action='store_true', default=True, help='Record video of the agent')
    parser.add_argument('--video-dir', type=str, default='videos', help='Directory for storing videos')

    args = parser.parse_args()

    ax1_pos, ax2_pos = [float(val) for val in args.param_coords.split(',')]

    full_run_name = [name for name in next(os.walk(f'{PROJECT_ROOT}/{args.exp_dir}'))[1] if args.run_name in name][0]
    run_path = f'{PROJECT_ROOT}/{args.exp_dir}/{full_run_name}/'
    final_model = [name for name in next(os.walk(run_path))[2] if 'model' in name][-1]

    opt_model = PPO.load(f'{PROJECT_ROOT}/{args.exp_dir}/{full_run_name}/{final_model}', device='cpu')
    deterministic = True
    if args.stdev != 0.0:
        opt_model.policy.set_std(args.stdev)
        deterministic = False

    env_kwargs = {}

    if 'pendulum' in args.env_name.lower():
        with open(f'{run_path}/description.txt') as desc_file:
            lines = desc_file.readlines()
            line = [l for l in lines if 'action_weight' in l]
            action_weight = float(line[0].split('=')[1].strip())
            env_kwargs['action_weight'] = action_weight
            env_kwargs['curriculum_coeff'] = 0
            print(f'Action weight: {action_weight}')
    elif 'cheetah' in args.env_name.lower():
        env_kwargs['assist_coeff'] = 0/3
    elif 'humanoid' in args.env_name.lower():
        with open(f'{run_path}/description.txt') as desc_file:
            lines = desc_file.readlines()
            line = [l for l in lines if 'control_cost_weight' in l]
            action_weight = float(line[0].split('=')[1].strip())
            env_kwargs['ctrl_cost_weight'] = action_weight
            print(f'Control cost weight: {action_weight}')
        env_kwargs['terminate_when_unhealthy'] = False
        env_kwargs['assist_coeff'] = 0/3
    else:
        raise ValueError('Unrecognized environment')

    env = make_vec_env(args.env_name, env_kwargs=env_kwargs)
    saved_envs = [name for name in next(os.walk(run_path))[2] if 'env' in name]
    env = VecNormalize.load(f'{run_path}/{saved_envs[-1]}', env)
    env.training = False
    env.norm_reward = False

    landscape_file = next(os.walk(f'{PROJECT_ROOT}/{args.landscape_dir}'))[2][0]
    data = np.load(f'{PROJECT_ROOT}/{args.landscape_dir}/{landscape_file}', allow_pickle=False)
    ax1, ax2 = data['ax1'], data['ax2']

    new_params_dict = compute_new_model_params(ax1, ax2, ax1_pos, ax2_pos, opt_model)
    opt_model.set_parameters(new_params_dict)

    obs = env.reset()
    episode_count = 0
    ep_return = []
    rew_sum = 0

    if args.record_video:
        images = []
        obs = env.reset()
        img = env.render(mode='rgb_array')
        for i in range(args.episodes * 200):
            images.append(img)
            action, _ = opt_model.predict(obs, deterministic=deterministic)
            obs, _, _, _ = env.step(action)
            img = env.render(mode='rgb_array')

        imageio.mimsave(f'{PROJECT_ROOT}/{args.video_dir}/{args.env_name}.gif',
                        [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=30)
    else:
        while episode_count < args.episodes:
            action, states = opt_model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, info = env.step(action)
            rewards = env.get_original_reward()
            rew_sum += rewards
            if not args.headless:
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
