import pickle
from utils.learning_algorithms import *
from environments.cliff import *
from environments.hole import *
from environments.umaze import *
from environments.random_maze import *
from environments.gym_utils import *


def start_experiment(environment, algos, num_episodes, test_freq, num_avg):

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
    elif environment == "umaze":  # select umaze
        env_id = "Umaze-v0"
        gym.envs.register(id=env_id, entry_point=UMaze, max_episode_steps=100)
        env = gym.make(env_id)
    else:
        raise ValueError("Please, select an available environment from the list in run.py")

    to_run = {"SCRN": discrete_SCRN, "SPG": discrete_policy_gradient, "SPG Entropy": discrete_policy_gradient,
              "Two stages SPG Entropy": discrete_policy_gradient}
    params = {"env": env, "num_episodes": num_episodes, "test_freq": test_freq}
    alphas = {"SCRN": 1e-3, "SPG": 1e-2, "SPG Entropy": 1e-2, "Two stages SPG Entropy": 1e-2}
    stats = {key: {} for key in algos}

    for i in range(num_avg):
        for algo in algos:
            algo_params = {**params, "alpha": alphas[algo]}
            algo_name = to_run[algo]
            if algo == "SPG Entropy" or algo == "Two stages SPG Entropy":
                algo_params = {**algo_params, "entropy_bonus": True}
            if algo == "Two stages SPG Entropy":
                two_phase_params = {"B1": 16, "B2": 1, "T": int(num_episodes/5)}
                algo_params = {**algo_params, "two_phases_params": two_phase_params}
            print(f"========== TRAINING RUN {i} OUT OF {num_avg} WITH {algo} ===========")
            current_stats = algo_name(**algo_params)
            stats[algo].update({i: current_stats})

    average_stats = {}
    std_stats = {}

    for key in algos:
        average_stats[key] = {item: np.mean([stats[key][i][item] for i in range(num_avg)], axis=0)
                              for item in ["steps", "rewards", "taus", "theta", "QOI", "obj_estimates", "grad_estimates"]}
        std_stats[key] = {item: np.std([stats[key][i][item] for i in range(num_avg)], axis=0) / np.sqrt(num_avg)
                          for item in ["steps", "rewards", "taus", "theta", "QOI", "obj_estimates", "grad_estimates"]}

    with open(f"new_results/{environment}_results.pkl", "wb") as handle:
        pickle.dump({"avg": average_stats, "std": std_stats}, handle, protocol=pickle.HIGHEST_PROTOCOL)


start_experiment("hole", ["SCRN", "SPG", "SPG Entropy", "Two stages SPG Entropy"],
                 num_episodes=10000, test_freq=50, num_avg=10)
