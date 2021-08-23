import torch
from custom_ppo import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from definitions import PROJECT_ROOT
import argparse
import numpy as np
from pendulum_env import PendulumEnv
from half_cheetah_v3_custom import HalfCheetahEnv
from humanoid_v3_custom import HumanoidEnv
import os
import multiprocessing
from sklearn.decomposition import PCA

"""
Slice visualization system for evaluating grids of points in optimization landscapes. This system can be used for
visualizing the effects of curriculum design and exploration noise on optimization landscapes. This system is based on
the implementation of Goldstein et al., available at https://github.com/tomgoldstein/loss-landscape.

This version of the system has two options for slice alignment: directly from start and endpoints, and PCA of recorded
snapshots (similar to Goldstein et al.). The first alignment method requires two sets of policy snapshots, while one
set is sufficient for PCA.

The workflow of this system is as follows:

    1.  Specify the curriculum and noise parameter values
    2.  Compute basis vectors for a slice from the recorded snapshots
    3.  Evaluate a grid of points centered on the origin. The position of each point is a linear combination of the
        basis vectors and the endpoint of the main training run (specified with parameters "--models-dir" and
        "--run-name") is the origin.
    4.  Project the optimization paths of the training runs onto the slice
    5.  Save the grid, the projected paths, and the basis vectors into a numpy archive (.npz)
    
    Repeat steps 1-5 for each required slice.
"""


def crunch(i_ind, j_ind, env_name, seed, base_path, curriculum, curriculum_component, stage_param, opt_params_flat,
           extent, n_eval, n_eps, ax1, ax2, device, stage_num, slice_stdev, cheetah_setpoint):
    """
    The main computation function. Loads the parameters of a point into an agent and runs test episodes. The average
    return of the episodes is the value of the point.

    :param i_ind: index to describe position in the direction of the first basis vector
    :param j_ind: index to describe position in the direction of the second basis vector
    :param env_name: identifier of the gym environment
    :param seed: seed for RNG
    :param base_path: path to the main training run
    :param curriculum: name of the curriculum
    :param curriculum_component: component of the curriculum, either "full", "noise", or "det" (just assistance)
    :param stage_param: parameter value for choosing the noise and curriculum settings, e.g. the level of assistance
    :param opt_params_flat: flat parameter vector of the model in the origin ("optimal" model)
    :param extent: distance from the origin to the edges of the grid. Span is 2*extent.
    :param n_eval: number of points per row (or column, the grid is square)
    :param n_eps: number of test episodes to run
    :param ax1: first basis vector
    :param ax2: second basis vector
    :param device: "cpu" or "cuda", used for setting tensors to the correct device
    :param stage_num: number specifying the point in the curriculum and the corresponding snapshot
    :param slice_stdev: standard deviation of the action noise for this slice
    :param cheetah_setpoint: stabilizer setpoint for the half-cheetah
    :return: average return of test episodes and index coordinates of the evaluated point
    """
    model = PPO.load(f'{base_path}/model_{stage_num}', device=device)

    # Set the specified parameters for the correct agent and environment
    if 'pendulum' in env_name.lower():
        with open(f'{base_path}/description.txt') as desc_file:
            lines = desc_file.readlines()
            line = [l for l in lines if 'action_weight' in l]
            action_weight = float(line[0].split('=')[1].strip())

        if curriculum not in ['assist', 'reward_shaping', 'reverse', 'stdev']:
            raise NotImplementedError(f'{curriculum} is not implemented for {env_name}')
        model.policy.set_std(slice_stdev)
        if curriculum_component != 'noise' and curriculum != 'stdev':
            curriculum_coeff = stage_param
        else:
            curriculum_coeff = 0
        env = make_vec_env(env_name, env_kwargs={'action_weight': action_weight, 'curriculum_type': curriculum,
                                                 'curriculum_coeff': curriculum_coeff})

    elif 'cheetah' in env_name.lower():
        if curriculum not in ['assist', 'stdev']:
            raise NotImplementedError(f'{curriculum} is not implemented for {env_name}')
        model.policy.set_std(slice_stdev)
        if curriculum_component != 'noise' and curriculum != 'stdev':
            assist_coeff = stage_param
        else:
            assist_coeff = 0
        env = make_vec_env(env_name, env_kwargs={'assist_coeff': assist_coeff, 'assist_setpoint': cheetah_setpoint})

    elif 'humanoid' in env_name.lower():
        if curriculum not in ['assist', 'stdev']:
            raise NotImplementedError(f'{curriculum} is not implemented for {env_name}')
        model.policy.set_std(slice_stdev)
        if curriculum_component != 'noise' and curriculum != 'stdev':
            assist_coeff = stage_param
        else:
            assist_coeff = 0
        env = make_vec_env(env_name, env_kwargs={'assist_coeff': assist_coeff, 'terminate_when_unhealthy': False})

    else:
        raise NotImplementedError(f'Unrecognized environment: {env_name}')

    if len([name for name in next(os.walk(base_path))[2] if 'env' in name]) > 1:
        env = VecNormalize.load(f'{base_path}/env_{stage_num}', env)
    else:
        env = VecNormalize.load(f'{base_path}/env', env)

    env.seed(seed)
    env.training = False
    env.norm_reward = False

    # Get the parameter vector of a point specified by i_ind, j_ind, and the basis vectors
    new_params_flat = opt_params_flat + extent * ((i_ind / (n_eval - 1) * 2 - 1) * ax1
                                                  + (j_ind / (n_eval - 1) * 2 - 1) * ax2)

    # Load the parameters into an agent
    new_params_dict = model.get_parameters()
    dim_counter = 0
    for key, value in new_params_dict['policy'].items():
        flat_val_dim = value.flatten().shape[0]
        new_params_dict['policy'][key] = np.reshape(new_params_flat[dim_counter:dim_counter+flat_val_dim], value.shape)
        new_params_dict['policy'][key] = torch.from_numpy(new_params_dict['policy'][key]).float().to(device)
        dim_counter += flat_val_dim

    model.set_parameters(new_params_dict)

    obs = env.reset()
    ep_rews = []
    rew_sum = 0

    if curriculum_component in ['noise', 'full']:
        deterministic = False
    else:
        deterministic = True

    # Run test episodes
    while len(ep_rews) < n_eps:
        action, states = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, info = env.step(action)
        rewards = env.get_original_reward()
        rew_sum += rewards
        if any(dones):
            ep_rews.append(rew_sum)
            rew_sum = 0

    mean_rews = np.mean(ep_rews)
    return mean_rews, i_ind, j_ind


def evaluate_points_grid(args, device, base_path, ax1, ax2, eval_stage=-1):
    """
    Evaluate n_eval x n_eval grid of points in an optimization landscape.

    :param args: input arguments
    :param device: "cpu" or "cuda"
    :param base_path: path to the main training run
    :param ax1: first basis vector
    :param ax2: second basis vector
    :param eval_stage: number specifying the point in the curriculum and the corresponding policy snapshot
    :return: grid of values, optimization paths of the main training run and the baseline run
    """
    model_names = [name for name in next(os.walk(base_path))[2] if 'model' in name]
    final_number = max([int(name.split('_')[1].split('.')[0]) for name in sorted(model_names)])

    # Get the parameter vector of the "optimal" policy
    opt_model_path = base_path + f'model_{final_number}'
    opt_params_dict = PPO.load(opt_model_path).get_parameters()
    opt_params_flat = np.concatenate([v.cpu().flatten() for v in opt_params_dict['policy'].values()])

    coeff = np.linspace(0, args.n_stages, args.n_stages)[eval_stage-1]
    coeff /= args.n_stages

    # Select curriculum and noise parameters
    assist_anneal_rate = 5/(4 * (args.n_stages-1))
    coeff = max(1 - (eval_stage-1) * assist_anneal_rate, 0)
    stdev_vals = [float(val) for val in args.stdev_range.split(',')]
    stdev_range = np.linspace(stdev_vals[0], stdev_vals[1], args.n_stages)
    slice_sd = stdev_range[eval_stage-1]

    values = np.zeros((args.n_eval, args.n_eval))

    # Main computation loop
    for i in range(args.n_eval):
        #if i == 0 or (i + 1) % 25 == 0:
        if args.verbose:
            print('==> %d of %d' % (i + 1, args.n_eval))

        with multiprocessing.Pool(processes=args.n_workers) as pool:
            arglist = [(i, j, args.env_name, args.seed, base_path, args.curriculum, args.curriculum_component, coeff,
                        opt_params_flat, args.extent, args.n_eval, args.eps_per_point, ax1, ax2, device, final_number,
                        slice_sd, args.cheetah_setpoint) for j in range(args.n_eval)]

            results = pool.starmap_async(crunch, arglist).get()

        for res in results:
            values[res[1], res[2]] = res[0]

    if args.verbose:
        print('%.2f => %.2f' % (np.min(values), np.max(values)))

    # Project optimization paths onto slice
    good_optim_path = get_projected_optimization_path(ax1, ax2, base_path, opt_params_flat)
    bad_optim_path = None
    if args.basis_method == 'model':
        baseline_path = f'{PROJECT_ROOT}/{args.baseline_dir}/'
        bad_optim_path = get_projected_optimization_path(ax1, ax2, baseline_path, opt_params_flat)

    return np.array([values]), good_optim_path, bad_optim_path


def generate_pca_basis(base_path):
    """
    Generate basis vectors by using PCA on the recorded policy snapshots

    :param base_path: path to the main training run
    :return: basis vectors ax1 and ax2: pair of 1xN_networkparams numpy arrays
    """
    model_names = [name for name in next(os.walk(base_path))[2] if 'model' in name]
    model_names = model_names[1:]

    snapshots = []

    for model in model_names:
        model_params_dict = PPO.load(f'{base_path}{model}').get_parameters()
        model_params_flat = np.concatenate([v.cpu().flatten() for v in model_params_dict['policy'].values()])
        snapshots.append(model_params_flat)

    snapshots = np.array(snapshots)
    pca = PCA(n_components=2)
    pca.fit(snapshots)

    ax1 = np.array(pca.components_[0])
    ax2 = np.array(pca.components_[1])

    return ax1, ax2


def generate_model_basis(curriculum_path, baseline_path):
    """
    Compute basis vectors from the locations of the initial model and the final models of the main training run and
    the baseline training run.

    :param curriculum_path: path to the curriculum training run (main training run)
    :param baseline_path: path to the baseline training run
    :return: basis vectors ax1 and ax2: pair of 1xN_networkparams numpy arrays
    """
    model_names = [name for name in next(os.walk(curriculum_path))[2] if 'model' in name]
    final_number = max([int(name.split('_')[1].split('.')[0]) for name in sorted(model_names)])

    baseline_model = [name for name in next(os.walk(baseline_path))[2] if 'model' in name][-1]

    initial_model_path = curriculum_path + f'model_0'
    opt_model_path = curriculum_path + f'model_{final_number}'
    baseline_opt_path = baseline_path + baseline_model

    initial_params_dict = PPO.load(initial_model_path).get_parameters()
    opt_params_dict = PPO.load(opt_model_path).get_parameters()
    baseline_params_dict = PPO.load(baseline_opt_path).get_parameters()

    initial_params_flat = np.concatenate([v.cpu().flatten() for v in initial_params_dict['policy'].values()])
    opt_params_flat = np.concatenate([v.cpu().flatten() for v in opt_params_dict['policy'].values()])
    baseline_params_flat = np.concatenate([v.cpu().flatten() for v in baseline_params_dict['policy'].values()])

    # Compute ax1
    ax1 = initial_params_flat - opt_params_flat
    ax1 /= np.linalg.norm(ax1)
    ax1 /= np.linalg.norm(ax1)

    # Compute ax2
    ax2 = baseline_params_flat - opt_params_flat
    ax2 = ax2 - np.dot(ax2, ax1) * ax1
    ax2 /= np.linalg.norm(ax2)

    return ax1, ax2


def generate_random_basis(seed, models_path):
    """
    Compute a random set of basis vectors.
    :param models_path: path to trained model
    :return: basis vectors ax1 and ax2: pair of 1xN_networkparams numpy arrays
    """
    model_names = [name for name in next(os.walk(models_path))[2] if 'model' in name]
    final_number = max([int(name.split('_')[1].split('.')[0]) for name in sorted(model_names)])
    final_params_dict = PPO.load(f'{models_path}model_{final_number}').get_parameters()
    final_params_flat = np.concatenate([v.cpu().flatten() for v in final_params_dict['policy'].values()])
    dim = final_params_flat.shape[0]

    rng = np.random.default_rng(seed)

    # Compute ax1
    ax1 = rng.uniform(-1, 1, dim)
    ax1 /= np.linalg.norm(ax1)

    # Compute ax2
    ax2 = rng.uniform(-1, 1, dim)
    ax2 = ax2 - np.dot(ax2, ax1) * ax1
    ax2 /= np.linalg.norm(ax2)

    return ax1, ax2


def get_projected_optimization_path(ax1, ax2, models_path, opt_params_flat):
    """
    Project the optimization path of a training run onto the slice defined by basis vectors ax1 and ax2.

    :param ax1: first basis vector
    :param ax2: second basis vector
    :param models_path: path to recorded snapshots
    :param opt_params_flat: flat parameter vector of the final model
    :return: projected locations of the snapshots
    """
    model_names = sorted([name for name in next(os.walk(models_path))[2] if 'model' in name])
    path = []
    for name in model_names:
        model_params_dict = PPO.load(f'{models_path}/{name}').get_parameters()
        model_params_flat = np.concatenate([v.cpu().flatten() for v in model_params_dict['policy'].values()])
        path.append(model_params_flat)

    ax1_proj_path = np.array([np.dot(p - opt_params_flat, ax1) for p in path])
    ax2_proj_path = np.array([np.dot(p - opt_params_flat, ax2) for p in path])
    projected_path = np.array([[p for p in ax1_proj_path], [p for p in ax2_proj_path]])

    return projected_path


def main():
    """
    Main function. Calls the computation functions to generate a pair of basis vectors and to evaluate a grid of points.
    Saves the results to a numpy archive (.npz).
    :return:
    """
    parser = argparse.ArgumentParser('Visualize the landscape of a policy')
    parser.add_argument('--models-dir', type=str, default='results/experiment', help='Directory of saved models')
    parser.add_argument('--run-name', type=str, default='run_1',
                        help='Name of the run')
    parser.add_argument('--curriculum', type=str, default='reward_shaping',
                        help='Curriculum type, available options:\n'
                             '\tPendulum: "assist", "reward_shaping", "reverse"\n'
                             '\tCheetah: "assist"\n'
                             '\tHumanoid: "assist"')
    parser.add_argument('--curriculum-component', type=str, default='full',
                        help='Specify which components of the curriculum will be used in visualization. By default,'
                             'both components (exploration noise and assistance mechanism) are used. Options: "full",'
                             '"noise", "assistance"')
    parser.add_argument('--stdev-range', type=str, default='0.5,0.01',
                        help='Range for the standard deviation of the exploration noise (2 comma-separated values)')
    parser.add_argument('--cheetah-setpoint', type=float, default=0, help='Stabilizer setpoint for Half-cheetah.')
    parser.add_argument('--output-name', type=str, default='output',
                        help='Name of the output files. Filenames will consist of this and the stage number.')
    parser.add_argument('--n-stages', type=int, default=6,
                        help='Number of stage policies to visualize. Selection will be evenly spaced and includes '
                             'the first and last policies. If the number is less than 2, only the final stage will '
                             'be visualized.')
    parser.add_argument('--vis-stages', type=str, default=None,
                        help='Specific stages to visualize, comma-separated list. If not specified, all stages will be'
                             'visualized sequentially. NOTE: n-stages must be specified based on total number of all'
                             'stages, not just the ones that will be visualized.')
    parser.add_argument('--env-name', type=str, default='CustomPendulum-v0', help='Environment id')
    parser.add_argument('--n-eval', type=int, default=20,
                        help='Number of evaluation iterations in x and y directions (total iterations n_eval^2)')
    parser.add_argument('--eps-per-point', type=int, default=20,
                        help='Number of test episodes to average for each evaluated point')
    parser.add_argument('--extent', type=int, default=100, help='Extent of evaluated plane in ax1 and ax2 directions.')
    parser.add_argument('--n-workers', type=int, default=6, help='Number of parallel processes for number crunching.')
    parser.add_argument('--seed', type=int, default=1, help='Seed for the numpy random engine, used for random slices')
    parser.add_argument('--basis-method', type=str, default='model',
                        help='Specify the method for selecting basis vectors:'
                             '"model" for aligning based on good optimum, bad optimum, and curriculum initial point'
                             '(must give path of baseline experiment relative to project root directory),'
                             '"pca" for aligning based on two main PCA components of recorded snapshots'
                             '"random" for sampling basis vectors from a uniform distribution')
    parser.add_argument('--baseline-dir', type=str, default='results/baseline/run_1',
                        help='Directory of baseline experiment for "model" alignment method'
                             '(relative to project root directory)')
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Toggle to use GPU in computations')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Toggle to be verbose and print messages during operation')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'

    # Path to models
    base_path = f'{PROJECT_ROOT}/{args.models_dir}/{args.run_name}_{args.curriculum}_curriculum/'
    num_models = len([name for name in next(os.walk(base_path))[2] if 'model' in name])
    stages = np.linspace(1, args.n_stages, args.n_stages, dtype=int)
    if args.vis_stages is not None:
        vis_stages = [int(stage)-1 for stage in args.vis_stages.split(',')]
        stages = stages[vis_stages]
    if args.n_stages < 2:
        stages = [num_models-1]

    # Generate basis vectors
    if args.basis_method == 'model':
        baseline_path = f'{PROJECT_ROOT}/{args.baseline_dir}/'
        ax1, ax2 = generate_model_basis(base_path, baseline_path)
    elif args.basis_method == 'pca':
        ax1, ax2 = generate_pca_basis(base_path)
    elif args.basis_method == 'random':
        ax1, ax2 = generate_random_basis(args.seed, base_path)
    else:
        raise NotImplementedError(f'Unrecognized method for computing basis vectors: {args.basis_method}')

    # Evaluate grid of points for all required slices
    for stage in stages:
        if args.verbose:
            print(f'Stage {np.where(stages == stage)[0][0]+1}/{len(stages)}')
        values, good_path, bad_path = evaluate_points_grid(args, device, base_path, ax1, ax2, eval_stage=stage)

        np.savez(f'{PROJECT_ROOT}/plots/data/{args.output_name}_{stage}',
                 values=values, ax1=ax1, ax2=ax2, good_path=good_path, bad_path=bad_path, extent=args.extent)


if __name__ == "__main__":
    if os.name == 'posix':
        multiprocessing.set_start_method('spawn')
    main()
