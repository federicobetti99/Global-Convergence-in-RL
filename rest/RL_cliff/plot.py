import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from RL_cliff.environment import env_to_text


def plot_rewards(
    sim_output
) -> None:
    """
    Visualizes rewards
    """
    sns.set_theme(style="whitegrid")
    # Set x-axis label
    positions = np.arange(0, len(sim_output.reward_cache[0]) / 10, 100)
    labels = np.arange(0, len(sim_output.reward_cache[0]), 1000)
    alg_colors = ['red', 'green', 'blue', 'purple', 'magenta']
    for i in range(len(sim_output.reward_cache)):
        mod = len(sim_output.reward_cache[i]) % 10
        mean_reward = np.mean(sim_output.reward_cache[i][mod:].reshape(-1, 10), axis=1)
        mod_var = len(sim_output.std_alg_reward[i]) % 10
        var_reward = np.mean(sim_output.std_alg_reward[i][mod_var:].reshape(-1, 10), axis=1)
        # ax=sns.lineplot(data=mean_reward, label=sim_output.name_cache[i])
        plt.plot([i for i in range(len(mean_reward))], mean_reward, color=alg_colors[i], label=sim_output.name_cache[i])
        plt.fill_between(range(0, len(mean_reward)), mean_reward - 0.3*var_reward, mean_reward + 0.3*var_reward, color=alg_colors[i], alpha=0.2)
        # ax.fill_between(range(0,len(mean_reward)), mean_reward - ,mean_reward +  ,color=alg_colors[i], alpha=0.2)
        # Plot graph
    
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    font = {'family': 'normal',
            'weight': 'bold'}

    plt.rc('font', **font)

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.xticks(positions, labels)
    plt.ylabel(r'\textbf{Y-AXIS}', fontsize=13)
    plt.xlabel(r'\textbf{X-AXIS}', fontsize=14)
    plt.ylabel('Avg. episode return')
    plt.xlabel('# episodes')
    plt.legend(loc="best")
    
    ax = plt.gca()
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black") 
    ax.spines["right"].set_color("black")
    ax.spines["left"].set_color("black")
    plt.xlim([-5, 1000])
    plt.show() 
    plt.grid(True)
    plt.savefig('rewards.pdf')
    plt.savefig('rewards.eps', format='eps')
    plt.clf()
    # plt.tight_layout()
    
    return


def plot_steps(
    sim_output,
) -> None:
    """
    Visualize number of steps taken
    """
    alg_colors = ['red', 'green', 'blue', 'purple', 'magenta']
    positions = np.arange(0, len(sim_output.step_cache[0]) / 10, 100)
    labels = np.arange(0, len(sim_output.step_cache[0]), 1000)

    sns.set_theme(style="whitegrid")

    for i in range(len(sim_output.step_cache)):
        mod = len(sim_output.step_cache[i]) % 10
        mean_step = np.mean(sim_output.step_cache[i][mod:].reshape(-1, 10), axis=1)
        print(mean_step.shape)
        mod_var = len(sim_output.std_alg_step[i]) % 10
        var_step = np.mean(sim_output.std_alg_step[i][mod_var:].reshape(-1, 10), axis=1)
        # min_step = np.mean(sim_output.sim_output_step_inst[i][mod_var:].reshape(-1, 10), axis=1)
        # ax=sns.lineplot(data=mean_step, label=sim_output.name_cache[i])
        plt.plot([i for i in range(len(mean_step))], mean_step, color=alg_colors[i], label=sim_output.name_cache[i])
        plt.fill_between(range(0, len(mean_step)), mean_step - 0.3*var_step, mean_step + 0.3*var_step, color=alg_colors[i], alpha=0.2)
        # ax.fill_between(range(0,len(mean_step)), min_step ,mean_step+0.3* var_step ,color=alg_colors[i], alpha=0.2)

    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20
    font = {'family': 'normal',
            'weight': 'bold'}

    plt.rc('font', **font)

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # Plot graph
    ax = plt.gca()
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black") 
    ax.spines["right"].set_color("black")
    ax.spines["left"].set_color("black")
    plt.ylabel(r'\textbf{Y-AXIS}', fontsize=13)
    plt.xlabel(r'\textbf{X-AXIS}', fontsize=14)

    plt.xticks(positions, labels)
    plt.ylabel("Avg. path length")
    plt.xlabel("# episodes")
    plt.legend(loc="best")

    plt.grid(True)
    plt.xlim([-5, 1000])
    plt.show()
    plt.savefig('steps.pdf')
    plt.savefig('steps.eps', format='eps')
    plt.clf()

    return


def console_output(
    sim_output,
    num_episodes: int,
) -> None:
    """Print path and key metrics in console"""
    for i in range(len(sim_output.env_cache)):
        env_str = env_to_text(sim_output.env_cache[i])

        print('=====', sim_output.name_cache[i], '=====')
        print("Action after {} iterations:".format(num_episodes), "\n")
        print(env_str, "\n")
        print("Number of steps:", int(sim_output.step_cache[i][-1]), "(best = 13)", "\n")
        print("Reward:", int(sim_output.reward_cache[i][-1]), "(best = -2)", "\n")

    return


def plot_path(
    sim_output,
) -> None:
    """Plot the latest paths as heatmap"""

    # Set values for cliff
    for i in range(len(sim_output.env_cache)):
        for j in range(1, 11):
            sim_output.env_cache[i][3, j] = -1

        ax = sns.heatmap(sim_output.env_cache[i], square=True, cbar=True, xticklabels=False, yticklabels=False)
        ax.set_title(sim_output.name_cache[i])
        plt.show()
        plt.savefig('paths.pdf')

    return None
