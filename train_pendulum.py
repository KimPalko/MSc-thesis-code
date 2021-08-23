from pendulum_env import PendulumEnv
import torch
import numpy as np
import argparse
from definitions import PROJECT_ROOT
from pathlib import Path
from custom_ppo import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from policy_networks import ActorCriticPolicy
import time
from shutil import copyfile
import os


"""
Training script for the pendulum.
"""


def train(args):
    """
    Train a pendulum agent using the specified training parameters and curriculum.
    :param args: script arguments
    :return: nothing
    """
    with open(f'{PROJECT_ROOT}/{args.hyperparam_file}') as params_file:
        lines = params_file.readlines()
        lines.pop(0)
        params = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in lines}

    if args.curriculum != 'stdev':
        curriculum_rate = 1/(4 / 5 * args.timesteps)

        env = make_vec_env(args.env_name, args.num_envs, env_kwargs={'action_weight': args.action_weight,
                                                                     'curriculum_coeff': 1.0,
                                                                     'curriculum_rate': curriculum_rate,
                                                                     'curriculum_type': args.curriculum})
    else:
        env = make_vec_env(args.env_name, args.num_envs, env_kwargs={'action_weight': args.action_weight})

    env = VecNormalize(env, gamma=float(params['gamma']), clip_reward=5)
    env.seed(args.seed)

    steps_per_env = int(int(params['batch_size']) // args.num_envs)
    minibatch_size = int(params['minibatch_size'])

    anneal_rate = args.timesteps / (steps_per_env * args.num_envs)

    curriculum = args.curriculum + '_curriculum'
    save_dir = f'{PROJECT_ROOT}/{args.results_dir}/{args.experiment_name}_{curriculum}'

    # Initialize a model. If a pre-initialized model is provided, load it. Otherwise, create a new model.
    if args.init_model is None:
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[16], vf=[16])], squash_output=True,
                             anneal_rate=anneal_rate)
        model = PPO(ActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=args.verbose, seed=args.seed,
                    learning_rate=float(params['learning_rate']), gamma=float(params['gamma']),
                    n_epochs=int(params['epochs']), clip_range=float(params['clip_epsilon']),
                    vf_coef=float(params['vf_coeff']), ent_coef=float(params['ent_coeff']),
                    clip_range_vf=float(params['vloss_clip_val']), max_grad_norm=float(params['max_grad_norm']),
                    gae_lambda=float(params['gae_lambda']), n_steps=steps_per_env, batch_size=minibatch_size,
                    tensorboard_log=f'{PROJECT_ROOT}/{args.results_dir}/{args.experiment_name}_{curriculum}',
                    device=args.device)
    else:
        model = PPO.load(args.init_model, env, verbose=args.verbose, seed=args.seed,
                         learning_rate=float(params['learning_rate']), gamma=float(params['gamma']),
                         n_epochs=int(params['epochs']), clip_range=float(params['clip_epsilon']),
                         vf_coef=float(params['vf_coeff']), ent_coef=float(params['ent_coeff']),
                         clip_range_vf=float(params['vloss_clip_val']), max_grad_norm=float(params['max_grad_norm']),
                         gae_lambda=float(params['gae_lambda']), n_steps=steps_per_env, batch_size=minibatch_size,
                         tensorboard_log=f'{PROJECT_ROOT}/{args.results_dir}/{args.experiment_name}_{curriculum}',
                         device=args.device)

    model.setup_snapshots(args.n_snapshots, int(args.timesteps / steps_per_env), save_dir)
    model.policy.anneal_rate = anneal_rate

    model.save(f'{save_dir}/model_0')
    env.save(f'{save_dir}/env_0')

    if args.verbose:
        print(f'Training {args.env_name} with PPO using continuous {args.curriculum} curriculum for '
              f'{int(args.timesteps)} total timesteps.')

    model.learn(total_timesteps=args.timesteps)
    final_number = int([f for f in next(os.walk(save_dir))[2] if 'model' in f][-1].split('.')[0].split('_')[-1])
    if final_number == args.n_snapshots:
        pass
    else:
        final_number += 1
    model.save(f'{save_dir}/model_{final_number}')
    env.save(f'{save_dir}/env_{final_number}')


def write_experiment_summary(args):
    """
    Write a summary of the training, i.e. save the used parameter values.
    :param args: script arguments
    :return: nothing
    """
    Path(f'{PROJECT_ROOT}/{args.results_dir}/{args.experiment_name}_{args.curriculum}').mkdir(parents=True,
                                                                                              exist_ok=True)
    with open(f'{PROJECT_ROOT}/{args.results_dir}/{args.experiment_name}_{args.curriculum}/description.txt',
              mode='w') as desc_file:
        args_dict = vars(args)
        sortednames = sorted(args_dict.keys(), key=lambda x: x.lower())
        lines = [f'{key} = {args_dict[key]}\n' for key in sortednames if key != 'info']
        if args.info is not None:
            lines.append('\n### Additional information ###\n')
            lines.append(f'{args.info}')
        desc_file.writelines(lines)


def main():
    """
    Parse script arguments and start training.
    """
    parser = argparse.ArgumentParser('Train an agent with PPO in the customized Pendulum environment')
    parser.add_argument('--experiment-name', type=str, default='pendulum_1', help='Name of the experiment')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print information during training')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory for storing results')
    parser.add_argument('--hyperparam-file', type=str, default='parameters/pendulum_ppo_params.txt',
                        help='Path to hyperparameter file')
    parser.add_argument('--timesteps', type=int, default=1e4, help='Total training timesteps')
    parser.add_argument('--env-name', type=str, default='CustomPendulum-v0', help='ID of the gym environment')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of environments to run in parallel')
    parser.add_argument('--action-weight', type=float, default=10.0, help='Weight coefficient for the action cost')
    parser.add_argument('--low-init-only', action='store_true', default=False,
                        help='Toggle to always initialize the pendulum to bottom position')
    parser.add_argument('--curriculum-stages', type=int, default=4,
                        help='Number of stages in curriculum. When value is 1 or less, no curriculum is used.')
    parser.add_argument('--curriculum', type=str, default='stdev',
                        help='Which curriculum to use. Available options: "reverse", "reward_shaping",'
                             '"stdev", "assist"')
    parser.add_argument('--continuous-curriculum', action='store_true', default=False,
                        help='Use a continuous curriculum. Specify number of snapshots to get intermediate models.')
    parser.add_argument('--n-snapshots', type=int, default=6,
                        help='Number of policy snapshots to save during training. Only for "stdev" and continuous'
                             'curricula.')
    parser.add_argument('--init-model', type=str, default=None, help='Path to pre-initialized model.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for RNG')
    parser.add_argument('--info', type=str, default='', help='Give additional information about the run')
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Toggle to use cuda device for training')

    args = parser.parse_args()
    args.init_model = None if args.init_model is None else f'{PROJECT_ROOT}/{args.init_model}'
    args.device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'

    args.seed += int(time.time())

    if args.curriculum in ['stdev', 'reverse', 'assist', 'reward_shaping']:
        train(args)
    else:
        raise NotImplementedError('The selected curriculum is not implemented.')

    args.curriculum += '_curriculum'
    write_experiment_summary(args)
    copyfile(f'{PROJECT_ROOT}/{args.hyperparam_file}',
             f'{PROJECT_ROOT}/{args.results_dir}/{args.experiment_name}_{args.curriculum}/hyperparameters.txt')


if __name__ == '__main__':
    main()
