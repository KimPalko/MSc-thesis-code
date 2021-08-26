# The Effect of Curriculum Design on Movement Optimization Landscapes
This repository contains the scripts used in the experiments of my [Master's thesis](https://aaltodoc.aalto.fi/handle/123456789/108263).

### Contents
[Introduction](#introduction)

[Setup](#setup)

[Training agents](#training-agents)

[Analyzing the training](#analyzing-the-training)

[Visualizing](#visualizing)

[Miscellaneous](#miscellaneous)


## Introduction
The scripts in this repository are used for training machine learning agents in movement-related tasks and for visualizing the loss landscapes of trained neural networks.
Agents are trained using reinforcement learning and multiple training curricula are available for each supported agent.
Currently, three agents are supported: a modified version of the OpenAI Gym Pendulum, the MuJoCo Half-cheetah, and the MuJoCo Humanoid (images shown below).

<img src="/images/pendulum_diagram_bg.png" alt="Pendulum" width="150"/> <img src="/images/half_cheetah.png" alt="Half-cheetah" width="400"/> <img src="/images/humanoid.png" alt="Humanoid" width="400"/>


## Setup
An Anaconda environment file is provided to facilitate installing the required Python libraries. The main libraries are listed below:

- Pytorch
- Tensorboard
- MuJoCo for Python
- OpenAI Gym
- Stable-Baselines 3
- Matplotlib
- Seaborn
- Pandas
- Numpy

## Training agents
Agents are trained using Proximal Policy Optimization from the Stable-Baselines 3 library. When training, a training curriculum can be selected with the appropriate 
command line options. The three agents have their proprietary curricula, which are not necessarily applicable to the other agents, even in principle. 
In addition to selecting the curricula, command line options are used for specifying where trained models are saved.

PPO hyperparameters are specified using separate files, which are found in the "parameters" directory.

Here are three example commands, one for each agent:

	python train_pendulum.py\
		--experiment-name pendulum_1\
		--results-dir results/pendulum\
		--timesteps 1000000\
		--num-envs 1\
		--action-weight 10.0\
		--n-snapshots 6\
		--curriculum reward_shaping\
		--seed 1234\
		--info "Pendulum example"
	

	python train_half_cheetah.py\
		--experiment-name cheetah_1\
		--results-dir results/halfcheetah\
		--timesteps 3000000\
		--num-envs 1\
		--curriculum assist\
		--n-snapshots 6\
		--seed 1234\
		--info "Half-cheetah example"


	python train_humanoid.py\
		--experiment-name humanoid_1\
		--results-dir results/humanoid\
		--timesteps 15000000\
		--control-cost-weight 1.25\
		--num-envs 1\
		--curriculum assist\
		--n-snapshots 6\
		--seed 1234\
		--info "Humanoid example"

For more information about the command line options, please refer to the descriptions of the options (available in the .py files or via the command line).

During the execution of a training script, a number of snapshots are saved to the results directory. Snapshots provide a means to observe the behavior of an agent
at different times during training. Once the execution of the script completes, a summary containing the used hyperparameters and the values given to each command
option is saved along with the final snapshot.

## Analyzing the training
If Tensorboard logging was enabled during training, performance graphs of the run can be plotted. The graphs are plotted using `plot_logs.py`. Tensorboard logs
are saved to the results directory, and the log files are named according to the experiment name and chosen curriculum.

This example command illustrates how the logging script is used:

	python plot_logs.py\
		--experiments-dir results/humanoid\
		--experiments "humanoid_1,humanoid_2,humanoid_3"\
		--legend-labels "Baseline;Curriculum_1;Curriculum_2"\
		--title "Training results"

As with the training scripts, more information about the command line options is available in `plot_logs.py`.

The format of the training performance graph is as follows:

<img src="/images/humanoid_training_multi.png" alt="humanoid_training" width="500"/>


## Visualizing
Landscape visualizations are made using scripts `generate_landscape_data.py` and `visualize_landscape.py`. The first computes a two-dimensional array of data
by evaluating the appropriate reward function at multiple points on a slice in the optimization landscape of a neural network. This slice is positioned by selecting
two basis vectors, either randomly or based on points of interest. Points of interest can be found optima and starting points of optimization, for example. The
points that are evaluated form a grid on the slice. The second script produces a contour plot based on the grid of evaluated points. The process of visualization
is explained in more detail in my thesis, and the following image summarizes the process:

<img src="/images/visualization_flowchart.png" alt="visualization_process" width="500"/>

The following figure shows an example of slice visualization set made with the visualization scripts:

<img src="/images/humanoid_landscapes_5x.jpeg" alt="humanoid_landscape" width="600"/>

The scripts support three options for choosing the basis vectors for slice alignment:

- Computing from points of interest
- Two-component PCA on the optimization path (saved snapshots)
- Random alignment

The first and second options were used in the thesis, and are recommended, because they are more reliably informative than random alignment.

Positioning slices based on points of interest requires three unique sets of network parameters. Training two agents starting from the same initial model but with
different RNG seeds conveniently provides these sets. The following example command shows how the data generator script can be used to produce landscape data when
the slice is aligned based on points of interest:

	python generate_landscape_data.py\
		--models-dir results/humanoid_curriculum\
		--run-name humanoid_1\
		--output-name example_visualization_1\
		--curriculum assist\
		--n-stages 6\
		--vis-stages 1\
		--basis-method model\
		--baseline-dir results/humanoid_baseline/humanoid_1\
		--env-name CustomHumanoid-v3\
		--n-eval 200\
		--eps-per-point 20\
		--extent 100\
		--n-workers 1\
		--seed=1234\
		--verbose

This example assumes that one humanoid agent has been trained using an assist curriculum and another without. Both training runs are assumed to have recorded six
snapshots. Here, the option `--vis-stages 1` tells the script to evaluate the landscape at the time of the first snapshot.

**Note**: the baseline directory must be the full path of a specific training run, whereas the "good" run is given by the model directory `--models-dir` and
the name of the run `--run-name`. Refer to `generate_landscape_data.py` for details on the command options.

Once the generator script completes, the produced data can be used to make a contour plot with `visualize_landscape.py`. The script can be used to plot contours
for multiple slices or just a single one. The configuration of the contour plots (rows and columns) can be adjusted with command line options.

**Use case example**: let us assume that two sets of slices have been computed, one for a curriculum training run, another for a training run without a curriculum.
Both sets of slices were computed with direct point-of-interest alignment. For correct operation, **the sets must be in separate directories**, otherwise all
contour plots will appear on the same row. The script automatically plots all slices in each directory. The following example command shows how to use
 `visualize_landscape.py` to create a figure with multiple slice sets:

	python visualize_landscape.py\
		--exp-dirs "humanoid_curriculum_slices,humanoid_pca_slices"\
		--plot-rows "0,1"\
		--plot-good-path\
		--plot-bad-path\
		--contour-levels 100\
		--zoom 5.0\
		--figure-title "Humanoid landscape slices"\
		--alignment model\
		--filter-sigma 1.0\
		--min -1000\
		--max 3000\
		--show-labels\
		--curriculum-quantity "Assistance level"\
		--quantity-range "100,0"\
		--unit "%"\
		--sigma-range "1.0,0.1"\
		--show-image


## Miscellaneous
The behaviors of trained agents can be tested using `test_model.py`.

Specific points in a landscape can be selected for testing; each point is a set of network parameters. These parameters can be loaded for testing with
`test_model_from_landscape.py`.

Untrained networks can be initialized with `pre_init_model.py`. This allows one to reuse an untrained agent for multiple training runs.

This codebase uses a modified version of the PPO and policy network scripts from Stable-Baselines 3: `policy_networks.py` and `custom_ppo.py` implement linearly
annealed standard deviation for exploration noise. This feature was unavailable in the original implementation.