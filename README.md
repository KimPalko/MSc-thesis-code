# The Effect of Curriculum Design on Movement Optimization Landscapes
This repository contains the scripts used in the experiments of my [Master's thesis](https://aaltodoc.aalto.fi/handle/123456789/108263).

### Contents
[Introduction](#introduction)

[Setup](#setup)

[Training agents](#training-agents)

[Analyzing the training](#analyzing-the-training)

[Visualizing](#visualizing)


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