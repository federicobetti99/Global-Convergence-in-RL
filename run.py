import pickle
from utils.training_utils import *
from utils.learning_algorithms import *
from environments.cliff import *
from environments.hole import *
from environments.umaze import *
from environments.random_maze import *
from environments.gym_utils import *


def start_experiment(environment, num_episodes, test_freq, num_avg):

    if environment == "random_maze":  # select random maze
        env_id = "RandomMaze-v0"
        gym.envs.register(id=env_id, entry_point=RandomMaze, max_episode_steps=100)
        env = gym.make(env_id)
    elif environment == "cliff":  # select cliff
        env_id = "RandomCliff-v0"
        gym.envs.register(id=env_id, entry_point=RandomCliff, max_episode_steps=100)
        env = gym.make(env_id)
    elif environment == "hole":  # select hole environment
        env_id = "Hole-v0"
        gym.envs.register(id=env_id, entry_point=RandomHole, max_episode_steps=100)
        env = gym.make(env_id)
    elif environment == "umaze":
        env_id = "Umaze-v0"
        gym.envs.register(id=env_id, entry_point=UMaze, max_episode_steps=100)
        env = gym.make(env_id)
    else:
        raise ValueError("Please, select an available environment from the list in run.py")

    stats = {"SCRN": {}, "SPG": {}, "SPG Entropy": {}, "Two stages SPG Entropy": {}}

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
        print("********** TRAINING WITH TWO STAGES regularized SPG *******")
        stats_DPG = discrete_policy_gradient(env, entropy_bonus=True, num_episodes=num_episodes,
                                             two_phases_params={"B1": 16, "B2": 1, "T": num_episodes/5},
                                             test_freq=test_freq)
        stats["Two stages SPG Entropy"].update({i: stats_DPG})

    average_stats = {"SCRN": {}, "SPG": {}, "SPG Entropy": {}, "Two stages SPG Entropy": {}}
    std_stats = {"SCRN": {}, "SPG": {}, "SPG Entropy": {}, "Two stages SPG Entropy": {}}

    average_stats["SCRN"] = {key: np.mean([stats["SCRN"][i][key] for i in range(num_avg)], axis=0)
                                    for key in ["steps", "rewards", "taus", "thetas", "obj_estimates", "grad_estimates"]}
    average_stats["SPG"] = {key: np.mean([stats["SPG"][i][key] for i in range(num_avg)], axis=0)
                                    for key in ["steps", "rewards", "taus", "thetas", "obj_estimates", "grad_estimates"]}
    average_stats["SPG Entropy"] = {key: np.mean([stats["SPG Entropy"][i][key] for i in range(num_avg)], axis=0)
                                    for key in ["steps", "rewards", "taus", "thetas", "obj_estimates", "grad_estimates"]}
    average_stats["Two stages SPG Entropy"] = {key: np.mean([stats["Two stages SPG Entropy"][i][key] for i in range(num_avg)], axis=0)
                                    for key in ["steps", "rewards", "taus", "thetas", "obj_estimates", "grad_estimates"]}
    std_stats["SCRN"] = {key: np.std([stats["SCRN"][i][key] for i in range(num_avg)], axis=0) / np.sqrt(num_avg)
                                    for key in ["steps", "rewards", "taus", "thetas", "obj_estimates", "grad_estimates"]}
    std_stats["SPG"] = {key: np.std([stats["SPG"][i][key] for i in range(num_avg)], axis=0) / np.sqrt(num_avg)
                                    for key in ["steps", "rewards", "taus", "thetas", "obj_estimates", "grad_estimates"]}
    std_stats["SPG Entropy"] = {key: np.std([stats["SPG Entropy"][i][key] for i in range(num_avg)], axis=0) / np.sqrt(num_avg)
                                    for key in ["steps", "rewards", "taus", "thetas", "obj_estimates", "grad_estimates"]}
    std_stats["Two stages SPG Entropy"] = {key: np.std([stats["Two stages SPG Entropy"][i][key] for i in range(num_avg)], axis=0) / np.sqrt(num_avg)
                                    for key in ["steps", "rewards", "taus", "thetas", "obj_estimates", "grad_estimates"]}

    with open(f"results/{environment}_results.pkl", "wb") as handle:
        pickle.dump({"avg": average_stats, "std": std_stats}, handle, protocol=pickle.HIGHEST_PROTOCOL)


start_experiment("hole", num_episodes=10000, test_freq=50, num_avg=10)
