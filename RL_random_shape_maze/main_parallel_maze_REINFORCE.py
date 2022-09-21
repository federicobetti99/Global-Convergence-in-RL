#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# import modules
from matplotlib import pyplot as plt
import pickle
from plot import plot_steps, plot_rewards, console_output, plot_path
from multiprocessing import Pool
from learning_algorithms import (
    discrete_policy_gradient,
)

import numpy as np

Total_Instances = 64
numthread = 16
Instances_per_Thread = int(Total_Instances / numthread)
num_episodes = 10000

run_algorithms = {
    "Discrete policy gradient",
}


def run_algs(ProcID):
    out_thread = []
    np.random.seed(ProcID)

    class sim_init:
        def __init__(self, num_episodes, gamma, alpha, epsilon, SGD, period):
            self.num_episodes = num_episodes  # Number of training episodes
            self.gamma = gamma  # Discount rate γ 0.9
            self.alpha = alpha  # Learning rate α 0.001
            self.epsilon = epsilon  # Exploration rate ε
            self.SGD = SGD
            self.l = 40
            self.rho = 200
            self.eps = 10 ** -3
            self.c_ = 1
            self.T_eps = 15
            self.batch_size = 1
            self.period = period

    class sim_output:
        def __init__(self, rewards_cache, step_cache, env_cache, name_cache, std_alg_reward, std_alg_step):
            self.reward_cache = rewards_cache  # list of rewards
            self.step_cache = step_cache  # list of steps
            self.env_cache = env_cache  # list of final paths
            self.name_cache = name_cache  # list of algorithm names
            self.std_alg_reward = std_alg_reward
            self.std_alg_step = std_alg_step

        def __str__(self):
            return "# episodes: " + str(self.num_episodes) + "gamma: " + str(self.gamma) \
                   + "alpha: " + str(self.alpha) + "epsilon: " + str(self.epsilon)

    for i in range(Instances_per_Thread):
        sim_output = sim_output(
            rewards_cache=[], step_cache=[], env_cache=[], name_cache=[], std_alg_reward=[], std_alg_step=[])
        REINFORCE_reward = []
        REINFORCE_step = []
        # Run discrete policy gradient
        if "Discrete policy gradient" in run_algorithms:
            sim_input = sim_init(num_episodes=num_episodes, gamma=0.99, alpha=0.002, epsilon=0, SGD=0, period=1000)
            all_probs, sim_output, temp_goal = discrete_policy_gradient(sim_input, sim_output)
            DPG_training_output = np.zeros(1)
            DPG_training_output = 1
            if DPG_training_output == 1:
                REINFORCE_step = sim_output.step_cache[0]
                REINFORCE_reward = sim_output.reward_cache[0]

        class sim_output:
            def __init__(self, rewards_cache, step_cache, env_cache, name_cache, std_alg_reward, std_alg_step):
                self.reward_cache = rewards_cache  # list of rewards
                self.step_cache = step_cache  # list of steps
                self.env_cache = env_cache  # list of final paths
                self.name_cache = name_cache  # list of algorithm names
                self.std_alg_reward = std_alg_reward
                self.std_alg_step = std_alg_step

        out_thread.append([REINFORCE_step, REINFORCE_reward, [DPG_training_output]])
    return out_thread


if __name__ == "__main__":
    """Learn cliff walking policies"""

    pool = Pool()  # Create a multiprocessing Pool
    data_inputs = list(range(numthread))
    with Pool(numthread) as p:
        out = p.map(run_algs, data_inputs)

    print('out:', out)

    SGD_step = np.zeros((1, num_episodes))
    SCRN_step = np.zeros((1, num_episodes))
    REINFORCE_step = np.zeros((1, num_episodes))

    SGD_reward = np.zeros((1, num_episodes))
    SCRN_reward = np.zeros((1, num_episodes))
    REINFORCE_reward = np.zeros((1, num_episodes))

    for i in range(len(out)):
        for j in range(len(out[0])):
            # REINFORCE
            if out[i][j][0]:
                REINFORCE_step += out[i][j][0] / np.sum(
                    np.sum(np.array(out).reshape(numthread, Instances_per_Thread, 3)[:, :, 2]))
            else:
                REINFORCE_step = REINFORCE_step

            if out[i][j][1]:
                REINFORCE_reward += out[i][j][1] / np.sum(
                    np.sum(np.array(out).reshape(numthread, Instances_per_Thread, 3)[:, :, 2]))
            else:
                REINFORCE_reward = REINFORCE_reward


    class sim_output:
        def __init__(self, rewards_cache, step_cache, env_cache, name_cache, std_alg_reward, std_alg_step):
            self.reward_cache = rewards_cache  # list of rewards
            self.step_cache = step_cache  # list of steps
            self.env_cache = env_cache  # list of final paths
            self.name_cache = name_cache  # list of algorithm names
            self.std_alg_reward = std_alg_reward
            self.std_alg_step = std_alg_step


    # print(SGD_step)
    out0 = []
    out1 = []
    for i in range(len(out)):
        for j in range(len(out[0])):
            if out[i][j][0]:
                out0.append(out[i][j][0])
            if out[i][j][1]:
                out1.append(out[i][j][1])

    sim_DPG_output_step_std = np.std(np.array(out0), axis=0)
    if not out0:
        sim_DPG_output_step_std = np.zeros(num_episodes)

    sim_DPG_output_reward_std = np.std(np.array(out1), axis=0)
    if not out1:
        sim_DPG_output_reward_std = np.zeros(num_episodes)

    with open('REINFORCE_step_maze.pkl', 'wb') as file:

        # A new file will be created
        pickle.dump(REINFORCE_step[0], file)

    with open('REINFORCE_reward_maze.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(REINFORCE_reward[0], file)
    with open('REINFORCE_step_maze_std.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(sim_DPG_output_step_std, file)
    with open('REINFORCE_reward_maze_std.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(sim_DPG_output_reward_std, file)

    # Plot output
    sim_out_total = sim_output(
        rewards_cache=[], step_cache=[], env_cache=[], name_cache=[], std_alg_step=[], std_alg_reward=[])
    sim_out_total.step_cache.append(REINFORCE_step[0])
    sim_out_total.reward_cache.append(REINFORCE_reward[0])
    sim_out_total.std_alg_step.append(sim_DPG_output_step_std)
    sim_out_total.std_alg_reward.append(sim_DPG_output_reward_std)
    sim_out_total.name_cache.append("REINFORCE")

    # print(SGD_step)
    # print(sim_out_total.__dict__)
    plot_steps(sim_out_total)
    plot_rewards(sim_out_total)
    print('REINFORCE_true_instances:',
          np.sum(np.sum(np.array(out).reshape(numthread, Instances_per_Thread, 3)[:, :, 2])))
    x = np.zeros(1)
    x[0] = np.sum(np.sum(np.array(out).reshape(numthread, Instances_per_Thread, 3)[:, :, 2]))
    plt.plot([i for i in range(1)], x, color='red', label="Successful_trajectories")
    plt.xlabel('i')
    plt.ylabel('true_instances')
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig('true_instances_maze_Reinforce.pdf')
