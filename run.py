import pickle
from utils.utils import *
import matplotlib.pyplot as plt

environment = "cliff"  # among random maze, cliff, hole or taxi

if environment == "taxi":  # select taxi environment from gym
    from taxi.training_utils import *
    from taxi.learning_algorithms import *
    env = gym.make("Taxi-v3")
elif environment == "maze":  # select random maze
    from mazes.training_utils import *
    from mazes.learning_algorithms import *
    from environments.random_maze import *
    env_id = "RandomMaze-v0"
    gym.envs.register(id=env_id, entry_point=RandomMaze, max_episode_steps=100)
    env = gym.make(env_id)
elif environment == "cliff":  # select cliff
    from mazes.training_utils import *
    from mazes.learning_algorithms import *
    from environments.cliff import *
    env_id = "RandomCliff-v0"
    gym.envs.register(id=env_id, entry_point=RandomCliff, max_episode_steps=100)
    env = gym.make(env_id)
elif environment == "hole":  # select hole environment
    from mazes.training_utils import *
    from mazes.learning_algorithms import *
    from environments.hole import *
    env_id = "Hole-v0"
    gym.envs.register(id=env_id, entry_point=RandomHole, max_episode_steps=100)
    env = gym.make(env_id)

train = True
test_freq = 50
num_episodes = 10000
num_avg = 10

if train:
    stats = {"SCRN": {}, "SPG": {}, "SPG Entropy": {}}
    for i in range(num_avg):
        print(f"========== TRAINING RUN {i} OUT OF {num_avg} ===========")
        print("********** TRAINING WITH SCRN **********")
        stats_SCRN = discrete_SCRN(env, num_episodes=num_episodes, test_freq=test_freq)
        stats["SCRN"].update({i: stats_SCRN})
        print("********** TRAINING WITH SPG ********")
        stats_DPG = discrete_policy_gradient(env, num_episodes=num_episodes, test_freq=test_freq)
        stats["SPG"].update({i: stats_DPG})
        print("********** TRAINING WITH regularized SPG ********")
        stats_DPG = discrete_policy_gradient(env, entropy_bonus=True, num_episodes=num_episodes, test_freq=test_freq)
        stats["SPG Entropy"].update({i: stats_DPG})
else:
    with open("results.pkl", "rb") as f:
        average_stats = pickle.load(f)

average_stats = {"SCRN": {}, "SPG": {}, "SPG Entropy": {}}
std_stats = {"SCRN": {}, "SPG": {}, "SPG Entropy": {}}
average_stats["SCRN"] = {key: np.mean([stats["SCRN"][i][key] for i in range(num_avg)], axis=0)
                         for key in ["steps", "rewards", "taus", "obj_estimates", "grad_estimates"]}
average_stats["SPG"] =  {key: np.mean([stats["SPG"][i][key] for i in range(num_avg)], axis=0)
                         for key in ["steps", "rewards", "taus", "obj_estimates", "grad_estimates"]}
average_stats["SPG Entropy"] = {key: np.mean([stats["SPG Entropy"][i][key] for i in range(num_avg)], axis=0)
                                for key in ["steps", "rewards", "taus", "obj_estimates", "grad_estimates"]}
std_stats["SCRN"] = {key: np.std([stats["SCRN"][i][key] for i in range(num_avg)], axis=0) / np.sqrt(num_avg)
                     for key in ["steps", "rewards", "taus", "obj_estimates", "grad_estimates"]}
std_stats["SPG"] = {key: np.std([stats["SPG"][i][key] for i in range(num_avg)], axis=0) / np.sqrt(num_avg)
                    for key in ["steps", "rewards", "taus", "obj_estimates", "grad_estimates"]}
std_stats["SPG Entropy"] = {key: np.std([stats["SPG Entropy"][i][key] for i in range(num_avg)], axis=0) / np.sqrt(num_avg)
                            for key in ["steps", "rewards", "taus", "obj_estimates", "grad_estimates"]}

plt.rcParams.update(plt.rcParamsDefault)
plt.figure()
QOI_SCRN = average_stats["SCRN"]["steps"]
STD_SCRN = std_stats["SCRN"]["steps"]
QOI_SPG = average_stats["SPG"]["steps"]
STD_SPG = std_stats["SPG"]["steps"]
QOI_ESPG = average_stats["SPG Entropy"]["steps"]
STD_ESPG = std_stats["SPG Entropy"]["steps"]
plt.plot(QOI_SCRN, label="SCRN")
plt.fill_between(np.arange(0, num_episodes), QOI_SCRN-STD_SCRN, QOI_SCRN+STD_SCRN, alpha=0.2)
plt.plot(QOI_SPG, label="SPG")
plt.fill_between(np.arange(0, num_episodes), QOI_SPG-STD_SPG, QOI_SPG+STD_SPG, alpha=0.2)
plt.plot(QOI_ESPG, label="SPG Entropy")
plt.fill_between(np.arange(0, num_episodes), QOI_ESPG-STD_ESPG, QOI_ESPG+STD_ESPG, alpha=0.2)
plt.legend(loc="best", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Number of episodes", fontsize=20)
plt.ylabel("Average episode length", fontsize=20)
plt.savefig(f"figures/{environment}/episode_lengths.png")

plt.figure()
QOI_SCRN = average_stats["SCRN"]["taus"]
STD_SCRN = std_stats["SCRN"]["taus"]
QOI_SPG = average_stats["SPG"]["taus"]
STD_SPG = std_stats["SPG"]["taus"]
QOI_ESPG = average_stats["SPG Entropy"]["taus"]
STD_ESPG = std_stats["SPG Entropy"]["taus"]
plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_SCRN, label="SCRN")
plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_SCRN-STD_SCRN, QOI_SCRN+STD_SCRN, alpha=0.2)
plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_SPG, label="SPG")
plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_SPG-STD_SPG, QOI_SPG+STD_SPG, alpha=0.2)
plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_ESPG, label="SPG Entropy")
plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_ESPG-STD_ESPG, QOI_ESPG+STD_ESPG, alpha=0.2)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="best", fontsize=15)
plt.xlabel("Number of episodes", fontsize=20)
plt.ylabel(r"$\frac{J(\theta^{*}) - J(\theta)}{\vert \vert \nabla J(\theta) \vert \vert}$", fontsize=20)
plt.savefig(f"figures/{environment}/taus.png")

plt.figure(figsize=(8, 5))
QOI_SCRN = average_stats["SCRN"]["rewards"]
STD_SCRN = std_stats["SCRN"]["rewards"]
QOI_SPG = average_stats["SPG"]["rewards"]
STD_SPG = std_stats["SPG"]["rewards"]
QOI_ESPG = average_stats["SPG Entropy"]["rewards"]
STD_ESPG = std_stats["SPG Entropy"]["rewards"]
plt.plot(QOI_SCRN, label="SCRN")
plt.fill_between(np.arange(0, num_episodes), QOI_SCRN-STD_SCRN, QOI_SCRN+STD_SCRN, alpha=0.2)
plt.plot(QOI_SPG, label="SPG")
plt.fill_between(np.arange(0, num_episodes), QOI_SPG-STD_SPG, QOI_SPG+STD_SPG, alpha=0.2)
plt.plot(QOI_ESPG, label="SPG Entropy")
plt.fill_between(np.arange(0, num_episodes), QOI_ESPG-STD_ESPG, QOI_ESPG+STD_ESPG, alpha=0.2)
plt.legend(loc="best", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Number of episodes", fontsize=20)
plt.ylabel("Reward during training", fontsize=20)
plt.savefig(f"figures/{environment}/rewards.png")

plt.figure(figsize=(8, 5))
QOI_SCRN = average_stats["SCRN"]["obj_estimates"]
STD_SCRN = std_stats["SCRN"]["obj_estimates"]
QOI_SPG = average_stats["SPG"]["obj_estimates"]
STD_SPG = std_stats["SPG"]["obj_estimates"]
QOI_ESPG = average_stats["SPG Entropy"]["obj_estimates"]
STD_ESPG = std_stats["SPG Entropy"]["obj_estimates"]
plt.plot(QOI_SCRN, label="SCRN")
plt.fill_between(np.arange(0, num_episodes), QOI_SCRN-STD_SCRN, QOI_SCRN+STD_SCRN, alpha=0.2)
plt.plot(QOI_SPG, label="SPG")
plt.fill_between(np.arange(0, num_episodes), QOI_SPG-STD_SPG, QOI_SPG+STD_SPG, alpha=0.2)
plt.plot(QOI_ESPG, label="SPG Entropy")
plt.fill_between(np.arange(0, num_episodes), QOI_ESPG-STD_ESPG, QOI_ESPG+STD_ESPG, alpha=0.2)
plt.legend(loc="best", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Number of episodes", fontsize=20)
plt.ylabel("Objective during training", fontsize=20)
plt.savefig(f"figures/{environment}/objectives.png")

plt.figure(figsize=(8, 5))
QOI_SCRN = average_stats["SCRN"]["grad_estimates"]
STD_SCRN = std_stats["SCRN"]["grad_estimates"]
QOI_SPG = average_stats["SPG"]["grad_estimates"]
STD_SPG = std_stats["SPG"]["grad_estimates"]
QOI_ESPG = average_stats["SPG Entropy"]["grad_estimates"]
STD_ESPG = std_stats["SPG Entropy"]["grad_estimates"]
plt.plot(QOI_SCRN, label="SCRN")
plt.fill_between(np.arange(0, num_episodes), QOI_SCRN-STD_SCRN, QOI_SCRN+STD_SCRN, alpha=0.2)
plt.plot(QOI_SPG, label="SPG")
plt.fill_between(np.arange(0, num_episodes), QOI_SPG-STD_SPG, QOI_SPG+STD_SPG, alpha=0.2)
plt.plot(QOI_ESPG, label="SPG Entropy")
plt.fill_between(np.arange(0, num_episodes), QOI_ESPG-STD_ESPG, QOI_ESPG+STD_ESPG, alpha=0.2)
plt.legend(loc="best", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Number of episodes", fontsize=20)
plt.ylabel("Gradient norms during training", fontsize=20)
plt.savefig(f"figures/{environment}/gradients.png")
