from matplotlib import pyplot as plt
from plot import plot_steps, plot_rewards, console_output, plot_path
from multiprocessing import Pool

import numpy as np
import pickle

num_episodes = 5000
SGD_step = np.zeros((1, num_episodes))
SCRN_step = np.zeros((1, num_episodes))
REINFORCE_step = np.zeros((1, num_episodes))

SGD_reward = np.zeros((1, num_episodes))
SCRN_reward = np.zeros((1, num_episodes))
REINFORCE_reward = np.zeros((1, num_episodes))


class sim_output:
    def __init__(self, rewards_cache, step_cache, env_cache, name_cache, std_alg_reward, std_alg_step):
        self.reward_cache = rewards_cache  # list of rewards
        self.step_cache = step_cache  # list of steps
        self.env_cache = env_cache  # list of final paths
        self.name_cache = name_cache  # list of algorithm names
        self.std_alg_reward = std_alg_reward
        self.std_alg_step = std_alg_step


# Open the file in binary mode
with open('REINFORCE_step_maze.pkl', 'rb') as file:
    # Call load method to deserialize
    REINFORCE_step[0] = pickle.load(file)
with open('REINFORCE_reward_maze.pkl', 'rb') as file:
    # Call load method to deserialize
    REINFORCE_reward[0] = pickle.load(file)
with open('SGD_step_maze.pkl', 'rb') as file:
    # Call load method to deserialize
    SGD_step[0] = pickle.load(file)
with open('SGD_reward_maze.pkl', 'rb') as file:
    # Call load method to deserialize
    SGD_reward[0] = pickle.load(file)
with open('SCRN_step_maze.pkl', 'rb') as file:
    # Call load method to deserialize
    SCRN_step[0] = pickle.load(file)
with open('SCRN_reward_maze.pkl', 'rb') as file:
    # Call load method to deserialize
    SCRN_reward[0] = pickle.load(file)
with open('SCRN_step_maze_std.pkl', 'rb') as file:
    # Call load method to deserialize
    sim_SCRN_output_step_std = pickle.load(file)
with open('SCRN_reward_maze_std.pkl', 'rb') as file:
    # Call load method to deserialize
    sim_SCRN_output_reward_std = pickle.load(file)
with open('SGD_step_maze_std.pkl', 'rb') as file:
    # Call load method to deserialize
    sim_SGD_output_step_std = pickle.load(file)
with open('SGD_reward_maze_std.pkl', 'rb') as file:
    # Call load method to deserialize
    sim_SGD_output_reward_std = pickle.load(file)

with open('REINFORCE_step_maze_std.pkl', 'rb') as file:
    # Call load method to deserialize
    sim_DPG_output_step_std = pickle.load(file)
with open('REINFORCE_reward_maze_std.pkl', 'rb') as file:
    # Call load method to deserialize
    sim_DPG_output_reward_std = pickle.load(file)

sim_out_total = sim_output(
    rewards_cache=[], step_cache=[], env_cache=[], name_cache=[], std_alg_step=[], std_alg_reward=[])
sim_out_total.step_cache.append(SGD_step[0])
sim_out_total.reward_cache.append(SGD_reward[0])
sim_out_total.std_alg_step.append(sim_SGD_output_step_std)
sim_out_total.std_alg_reward.append(sim_SGD_output_reward_std)
sim_out_total.name_cache.append("SPG")

sim_out_total.step_cache.append(SCRN_step[0])
sim_out_total.reward_cache.append(SCRN_reward[0])
sim_out_total.std_alg_step.append(sim_SCRN_output_step_std)
sim_out_total.std_alg_reward.append(sim_SCRN_output_reward_std)
sim_out_total.name_cache.append("SCRN")

sim_out_total.step_cache.append(REINFORCE_step[0])
sim_out_total.reward_cache.append(REINFORCE_reward[0])
sim_out_total.std_alg_step.append(sim_DPG_output_step_std)
sim_out_total.std_alg_reward.append(sim_DPG_output_reward_std)
sim_out_total.name_cache.append("REINFORCE")

# print(SGD_step)
# print(sim_out_total.__dict__)
plot_steps(sim_out_total)
plot_rewards(sim_out_total)
