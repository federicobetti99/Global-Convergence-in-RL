import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.training_utils import *
from utils.learning_algorithms import *
from environments.cliff import *
from environments.hole import *
from environments.umaze import *
from environments.random_maze import *
from environments.gym_utils import *
import imageio


def compare_probabilities_learned(average_stats, state, save_link=None):
    """
    Compare probabilities learned during training
    :param average_stats: average stats collected during training
    :param state: state of observation space
    :param save_link: link where to save image
    """

    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 5))
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)

    fig.suptitle(f"Probabilities for state {state}")

    length = average_stats["SPG"]["history_probs"].shape[0]

    ax[0].plot([average_stats["SPG"]["history_probs"][j][state, 0] for j in range(length)], label="SPG - P(up)")
    ax[0].plot([average_stats["SPG"]["history_probs"][j][state, 1] for j in range(length)], label="SPG - P(down)")

    ax[1].plot([average_stats["SCRN"]["history_probs"][j][state, 0] for j in range(length)], label="SCRN - P(up)")
    ax[1].plot([average_stats["SCRN"]["history_probs"][j][state, 1] for j in range(length)], label="SCRN - P(down)")

    ax[0].plot([average_stats["SPG"]["history_probs"][j][state, 2] for j in range(length)], label="SPG - P(left)")
    ax[0].plot([average_stats["SPG"]["history_probs"][j][state, 3] for j in range(length)], label="SPG - P(right)")

    ax[1].plot([average_stats["SCRN"]["history_probs"][j][state, 2] for j in range(length)], label="SCRN - P(left)")
    ax[1].plot([average_stats["SCRN"]["history_probs"][j][state, 3] for j in range(length)], label="SCRN - P(right)")

    ax[0].legend(loc="best")
    ax[1].legend(loc="best")

    ax[0].set_ylabel(r"$\pi_{\theta} (\cdot \vert s)$", fontsize=20)
    ax[0].set_xlabel("Episode", fontsize=20)

    ax[1].set_ylabel(r"$\pi_{\theta} (\cdot \vert s)$", fontsize=20)
    ax[1].set_xlabel("Episode", fontsize=20)

    ax[0].set_yticks(np.linspace(0, 1, 6), fontsize=15)
    ax[0].set_xticks(np.linspace(0, length, 6), fontsize=15)
    ax[1].set_yticks(np.linspace(0, 1, 6), fontsize=15)
    ax[1].set_xticks(np.linspace(0, length, 6), fontsize=15)

    if save_link is not None:
        plt.savefig(save_link)

    plt.show()

    return fig, ax


def show_agent_position(env, position):
    """
    Show trajectory followed by agent
    :param env: environment
    :param position: current agent position
    :return:
    """
    plt.figure(figsize=(6, 3))
    sizes = env.maze.size
    plt.yticks(np.arange(0, sizes[0]), [])
    plt.xticks(np.arange(0, sizes[1]), [])
    for i in range(sizes[0]):
        plt.axhline(y=i+0.5, xmin=0, xmax=sizes[1])
    for j in range(sizes[1]):
        plt.axvline(x=j+0.5, ymin=0, ymax=sizes[0])
    twod_position = (int(position / sizes[0]), position % sizes[1])
    plt.plot([twod_position[0]+0.5], [twod_position[1]+0.5], label="o")
    plt.legend()
    plt.show()


def mark_path(agent: tuple, env: np.array) -> np.array:
    """
    Store path taken by agent
    Only needed for visualization
    :param agent: agent encoding
    :param env: environment
    :return:
        - path followed by agent
    """
    (posY, posX) = agent
    env[posY][posX] += 1

    return env


def env_to_text(env: np.array) -> str:
    """
    Convert environment to text format
    Needed for visualization in console
    :param env: environment
    """
    env = np.where(env >= 1, 1, env)
    env = np.array2string(env, precision=0, separator=" ", suppress_small=False)
    env = env.replace("[[", " |")
    env = env.replace("]]", "|")
    env = env.replace("[", "|")
    env = env.replace("]", "|")
    env = env.replace("1", "x")
    env = env.replace("0", " ")

    return env


def running_mean(x, n):
    """
    Running average of vector x values with window size n
    :param x: vector of values
    :param n: window size
    :return:
        - running average of x every n elements, for smoothing out rewards average over training
    """
    running_avg = [np.mean(x[i*n:(i+1)*n]) for i in range(int(len(x)/n))]
    return np.array(running_avg)


def plot_stats(average_stats, std_stats, num_episodes, test_freq):
    """
    Main function to plot statistics, produces 4 subplots with average episode length, reward, objective and gradient norm
    and a bigger plot for PL constant results on average
    :param average_stats: average stats for all RL algorithms
    :param std_stats: standard deviations for all RL algorithms
    :param num_episodes: number of training episodes
    :param test_freq: test frequency during training for PL inequality testing
    :return:
    """
    legend_keys = {"steps": "Episode length",
                   "taus": r"$\frac{J^\lambda(\theta^{*}) - J^\lambda(\theta)}{\| \| \nabla J^\lambda(\theta) \| \|^{\alpha}}$",
                   "rewards": "Episode reward",
                   "obj_estimates": r"$J(\theta)$",
                   "grad_estimates": r"$\| \| \nabla J(\theta) \| \|$",
                   "QOI": r"$\min_{s} \,\, \pi_{\theta_t}(a^*(s) \vert s)$",
                   "QOI Entropy": r"$\min_{s, a} \,\, \pi_{\theta_t}(a \vert s)$",
                   "eigs": r"$\lambda_{\min}(\nabla^2 J^\lambda(\theta) \mapsto \lambda_{\max}(\nabla^2 J^\lambda(\theta)$"}

    legend_items = {"SCRN": "SCRN", "SPG": "SPG", "SPG Entropy": "ESPG", "Two stages SPG Entropy": "TSESPG"}
    alphas = {"SCRN": 1, "SPG": 1, "SPG Entropy": 2, "Two stages SPG Entropy": 2}
    lambdas = {"SCRN": 0, "SPG": 0, "SPG Entropy": 1, "Two stages SPG Entropy": 1}
    legend_items_taus = {key: rf"{legend_items[key]}, $\alpha = {alphas[key]}$" for key in legend_items.keys()}
    steps = {"steps": int(test_freq/5), "taus": test_freq, "rewards": int(test_freq/5),
             "obj_estimates": test_freq, "grad_estimates": test_freq, "QOI": 1}

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    fig.tight_layout(pad=3.)
    fig.subplots_adjust(top=0.7, left=0.001, right=1.2, bottom=0.02, wspace=0.5, hspace=0.5)

    for ind, key in enumerate(["steps", "obj_estimates", "rewards", "grad_estimates"]):
        for algo in ["SCRN", "SPG", "SPG Entropy", "Two stages SPG Entropy"]:
            QOI = average_stats[algo][key]
            STD_QOI = std_stats[algo][key]
            if key in ["steps", "rewards"]:
                QOI = running_mean(QOI, steps[key])
                STD_QOI = running_mean(STD_QOI, steps[key])
            ax[ind % 2, int(ind/2)].plot(np.arange(0, num_episodes, step=steps[key]), QOI, label=legend_items[algo])
            ax[ind % 2, int(ind/2)].fill_between(np.arange(0, num_episodes, step=steps[key]), QOI-STD_QOI, QOI+STD_QOI, alpha=0.2)
            ax[ind % 2, int(ind/2)].set_xlabel("Number of episodes", fontsize=15)
            ax[ind % 2, int(ind/2)].tick_params(axis='both', which='major', labelsize=15)
            ax[ind % 2, int(ind/2)].tick_params(axis='both', which='minor', labelsize=15)
            if key in ["obj_estimates", "grad_estimates"]:
                ax[ind % 2, int(ind/2)].set_ylabel(legend_keys[key], fontsize=20)
            else:
                ax[ind % 2, int(ind/2)].set_ylabel(legend_keys[key], fontsize=15)

    plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.25), fancybox=True, shadow=True, ncol=4, fontsize='xx-large')

    plt.figure(figsize=(8, 5))
    key = "taus"
    for algo in ["SCRN", "SPG", "SPG Entropy", "Two stages SPG Entropy"]:
        QOI = average_stats[algo][key]
        STD_QOI = std_stats[algo][key]
        plt.plot(np.arange(0, num_episodes, step=steps[key]), QOI, label=legend_items_taus[algo])
        plt.fill_between(np.arange(0, num_episodes, step=steps[key]), QOI-STD_QOI, QOI+STD_QOI, alpha=0.2)
        plt.xlabel("Number of episodes", fontsize=15)
        plt.ylabel(legend_keys[key], fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.yscale("log")
        plt.legend(loc='upper center', bbox_to_anchor=(1.40, 0.8), fancybox=True, shadow=True, ncol=1, fontsize='xx-large')

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    fig.tight_layout(pad=3.)
    fig.subplots_adjust(top=0.7, left=0.001, right=1.2, bottom=0.02, wspace=0.5, hspace=0.5)
    key = "QOI"
    for algo in ["SCRN", "SPG"]:
        QOI = average_stats[algo][key]
        STD_QOI = std_stats[algo][key]
        ax[0].plot(np.arange(0, num_episodes, step=steps[key]), QOI, label=legend_items[algo])
        ax[0].fill_between(np.arange(0, num_episodes, step=steps[key]), QOI-STD_QOI, QOI+STD_QOI, alpha=0.2)
        ax[0].set_xlabel("Number of episodes", fontsize=15)
        ax[0].set_ylabel(legend_keys["QOI"], fontsize=20)
        ax[0].tick_params(axis='both', which='major', labelsize=15)
        ax[0].set_yscale("log")
        ax[0].legend(loc='upper center', bbox_to_anchor=(0.55, -0.2), fancybox=True, shadow=True, ncol=2, fontsize='xx-large')

    for algo in ["SPG Entropy", "Two stages SPG Entropy"]:
        QOI = average_stats[algo][key]
        STD_QOI = std_stats[algo][key]
        ax[1].plot(np.arange(0, num_episodes, step=steps[key]), QOI, label=legend_items[algo])
        ax[1].fill_between(np.arange(0, num_episodes, step=steps[key]), QOI-STD_QOI, QOI+STD_QOI, alpha=0.2)
        ax[1].set_xlabel("Number of episodes", fontsize=15)
        ax[1].set_ylabel(legend_keys["QOI Entropy"], fontsize=20)
        ax[1].tick_params(axis='both', which='major', labelsize=15)
        ax[1].set_yscale("log")
        ax[1].legend(loc='upper center', bbox_to_anchor=(0.55, -0.2), fancybox=True, shadow=True, ncol=2, fontsize='xx-large')


def final_trajectory(environment, theta):
    """
    Shows the final trajectory of training followed by a RL agent, saves snapshots of the environment
    for future postprocessing and GIF creations
    :param environment: environment string
    :param theta: parameter policy
    :return:
    """
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

    env.reset()
    env.render()

    state_trajectory = []
    count = 0

    while not env.end:
        # Get state corresponding to agent position
        state = env.get_state()

        screen = env.render(mode="human")
        plt.imsave(f"figures/trajectories/{environment}/{count}.png", screen)

        # Get probabilities per action from current policy
        action_probs = pi(env, theta)

        # Select greedy action according to policy
        action = np.argmax(np.squeeze(action_probs))

        # Move agent to next position
        next_state, reward, _, _, _ = env.step(action)

        state_trajectory.append(state)
        count += 1

    return count
