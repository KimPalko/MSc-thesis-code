# The Effect of Curriculum Design on Movement Optimization Landscapes
This repository contains the scripts used in the experiments of my [Master's thesis](https://aaltodoc.aalto.fi/handle/123456789/108263).

### Contents
[Introduction](#introduction)

[Setup](#setup)

[Training agents](#training-agents)

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


## Visualizing