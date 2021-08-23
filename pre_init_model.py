from pendulum_env import PendulumEnv
from half_cheetah_v3_custom import HalfCheetahEnv
from humanoid_v3_custom import HumanoidEnv
import torch
import argparse
from definitions import PROJECT_ROOT
from custom_ppo import PPO
from stable_baselines3.common.env_util import make_vec_env
from policy_networks import ActorCriticPolicy


"""
Script for initializing models. Useful when multiple training runs are required to start from the same model.
"""


def init_model(args):
    """
    Initialize a model based on the script arguments.
    :param args: script arguments
    :return: nothing
    """
    env = make_vec_env(args.env_name, args.num_envs)

    if 'pendulum' in args.env_name.lower():
        if args.net_archs is None:
            policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[16], vf=[16])], squash_output=True)
        else:
            actor_arch = [int(neurons) for neurons in args.net_archs.split(';')[0].split(',') if int(neurons) != 0]
            critic_arch = [int(neurons) for neurons in args.net_archs.split(';')[1].split(',') if int(neurons) != 0]
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                                 net_arch=[dict(pi=actor_arch, vf=critic_arch)], squash_output=True)
    elif 'cheetah' in args.env_name.lower():
        policy_kwargs = dict()
    elif 'humanoid' in args.env_name.lower():
        policy_kwargs = dict()
    else:
        raise NotImplementedError(f'The environment "{args.env_name}" is not available.')

    model = PPO(ActorCriticPolicy, env, policy_kwargs=policy_kwargs, device=args.device)

    model.save(f'{args.save_dir}/{args.model_name}')


def main():
    """
    Parse script arguments and call the initializer function.
    """
    parser = argparse.ArgumentParser('Pre-initialize a model for training')
    parser.add_argument('--env-name', type=str, default='CustomPendulum-v0',
                        help='Gym environment id')
    parser.add_argument('--net-archs', type=str, default=None,
                        help='Architectures for actor and critic nets, only for pendulum. Format: A1,A2,A3;C1,C2,C3'
                             'where A1-A3 and C1-C3 are integers. Use "0;0" for linear policy.')
    parser.add_argument('--save-dir', type=str, default='results/pre_init_models', help='Where to save the model.')
    parser.add_argument('--model-name', type=str, default='pendulum_init_model', help='Name of the model')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Toggle to use cuda device')

    args = parser.parse_args()

    args.device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'

    args.save_dir = f'{PROJECT_ROOT}/{args.save_dir}'
    if args.env_name in ['CustomPendulum-v0', 'CustomHalfCheetah-v3', 'CustomHumanoid-v3']:
        init_model(args)
    else:
        raise NotImplementedError(f'The environment "{args.env_name}" is not available.')


if __name__ == '__main__':
    main()
