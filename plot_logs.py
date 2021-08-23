import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from definitions import PROJECT_ROOT
import matplotlib.pyplot as plt
import argparse
import os


"""
Script for plotting training performance graphs. Reads data from Tensorboard log files.
"""


def plot_multiple(experiments, args):
    """
    Read the Tensorboard logs of all runs in the specified experiment directories.
    :param experiments: list of directories to experiments
    :param args: script arguments
    :return: nothing
    """
    results = []
    steps = np.array([])

    for exp in experiments:
        print(f'Experiment {experiments.index(exp)+1} of {len(experiments)}')
        runs = next(os.walk(exp))[1]
        run_results = []
        for run in runs:
            print(f'\tRun {runs.index(run) + 1} of {len(runs)}')
            stages = next(os.walk(f'{exp}/{run}'))[1]
            stage_results = np.array([])
            stage_steps = np.array([])
            last_step = 0

            path = f'{exp}/{run}'

            for stage in stages:
                event_acc = EventAccumulator(f'{path}/{stage}')
                event_acc.Reload()
                w_times, step_nums, vals = zip(*event_acc.Scalars('rollout/ep_rew_mean'))
                #w_times, step_nums, vals = zip(*event_acc.Scalars('train/std'))
                stage_steps = np.concatenate((stage_steps, np.array(step_nums) + last_step))
                last_step = stage_steps[-1]
                vals = np.array(vals)
                stage_results = np.concatenate((stage_results, vals))

            if len(steps) == 0 or len(stage_steps) < len(steps):
                steps = stage_steps
            run_results.append(stage_results)

        mean = np.mean(run_results, axis=0)
        stddev = np.std(run_results, axis=0)
        results.append((mean, stddev))

    n_steps = len(steps)
    plt.figure()

    colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:purple']
    for color_index, run in enumerate(results):
        plt.fill_between(steps, run[0][:n_steps] - run[1][:n_steps], run[0][:n_steps] + run[1][:n_steps],
                         alpha=0.35, color=colors[color_index])
        plt.plot(steps, run[0][:n_steps], color=colors[color_index])

    if args.legend_labels is not None:
        legend_labels = args.legend_labels.split(';')
        plt.legend(legend_labels, loc='lower right')

    plt.xlabel('Timesteps')
    plt.ylabel('Average episodic return')
    plt.title(args.title)
    plt.tight_layout()
    plt.show()


def main():
    """
    Parse script arguments and call the plotting function.
    """
    parser = argparse.ArgumentParser('Visualize Stable-Baselines training performance graphs')
    parser.add_argument('--experiments-dir', type=str, default='results/humanoid',
                        help='Directory of experiments to plot')
    parser.add_argument('--experiments', type=str,
                        default='humanoid_10_1,'
                                'humanoid_10_1_a',
                                #'cheetah_a2,'
                                #'cheetah_a3',
                        help='Directories of specific experiments. Comma-separated list of names.')
    parser.add_argument('--legend-labels', type=str,
                        default='Baseline;'
                                'Curriculum',
                                #'Curriculum a2;'
                                #'Curriculum a3',
                        help='Semicolon-separated legend entires. The order of the entries corresponds directly to'
                             'the order of the experiments.')
    parser.add_argument('--title', type=str, default='',
                        help='Title of the plot')
    args = parser.parse_args()
    #args.legend_labels = None
    #args.baseline_exp = None

    exp_dirs = args.experiments.split(',')
    experiments = [f'{PROJECT_ROOT}/{args.experiments_dir}/{exp}' for exp in exp_dirs]

    print(f'Plotting results of {len(experiments)} experiment{"s" if len(experiments) > 1 else ""}.')
    plot_multiple(experiments, args)


if __name__ == '__main__':
    main()
