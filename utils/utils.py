import numpy as np
import matplotlib.pyplot as plt


def compare_probabilities_learned(average_stats, state, save_link=None):
    """
    Compare probabilities learned during training
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


def show_trajectory(env, state_trajectory):
    """
    Show trajectory followed by agent
    :param env: environment
    :param state_trajectory: states visited during trajectory
    :return:
    """
    states = []
    plt.figure(figsize=(6, 3))
    plt.yticks(np.arange(0, 4), [])
    plt.xticks(np.arange(0, 12), [])
    for i in range(4):
        plt.axhline(y=0.2 * i, xmin=0, xmax=0.3 * 12)
    for j in range(12):
        plt.axvline(x=0.2 * j, ymin=0, ymax=0.3 * 4)
    for state in state_trajectory:
        state = env.state_to_position(state)
        states.append(state)
        plt.plot(0.2 * state[0], 0.2 * state[1], marker='X', color="g", markersize=10)
        if len(states) >= 2:
            plt.arrow(0.2 * states[-2][0], 0.2 * states[-2][1],
                      0.2 * (states[-1][0]-states[-2][0]), 0.2 * (states[-1][1]-states[-2][1]))
    plt.show()


def mark_path(agent: tuple, env: np.array) -> np.array:
    """
    Store path taken by agent
    Only needed for visualization
    """
    (posY, posX) = agent
    env[posY][posX] += 1

    return env


def env_to_text(env: np.array) -> str:
    """
    Convert environment to text format
    Needed for visualization in console
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


def running_average(arr):
    """
    Computes the running average of the quantities stored in arr
    :param arr: array
    :return:
        - the running average (computed after each item) of arr
    """
    return [np.sum(arr[:i]) / (i+1) for i in range(len(arr))]


def plot_stats(environment, average_stats, std_stats, num_episodes, test_freq):
    plt.figure()
    QOI_SCRN = average_stats["SCRN"]["steps"]
    STD_SCRN = std_stats["SCRN"]["steps"]
    QOI_SPG = average_stats["SPG"]["steps"]
    STD_SPG = std_stats["SPG"]["steps"]
    QOI_ESPG = average_stats["SPG Entropy"]["steps"]
    STD_ESPG = std_stats["SPG Entropy"]["steps"]
    QOI_TSESPG = average_stats["Two stages SPG Entropy"]["steps"]
    STD_TSESPG = std_stats["Two stages SPG Entropy"]["steps"]
    plt.plot(np.arange(0, num_episodes), QOI_SCRN, label="SCRN")
    plt.fill_between(np.arange(0, num_episodes), QOI_SCRN-STD_SCRN, QOI_SCRN+STD_SCRN, alpha=0.2)
    plt.plot(np.arange(0, num_episodes), QOI_SPG, label="SPG")
    plt.fill_between(np.arange(0, num_episodes), QOI_SPG-STD_SPG, QOI_SPG+STD_SPG, alpha=0.2)
    plt.plot(np.arange(0, num_episodes), QOI_ESPG, label="ESPG")
    plt.fill_between(np.arange(0, num_episodes), QOI_ESPG-STD_ESPG, QOI_ESPG+STD_ESPG, alpha=0.2)
    plt.plot(np.arange(0, num_episodes), QOI_TSESPG, label="TSESPG")
    plt.fill_between(np.arange(0, num_episodes), QOI_TSESPG-STD_TSESPG, QOI_ESPG+STD_TSESPG, alpha=0.2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.55, -0.2), fancybox=True, shadow=True, ncol=4, fontsize='large')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Number of episodes", fontsize=20)
    plt.ylabel("Average episode length", fontsize=20)

    plt.figure()
    QOI_SCRN = average_stats["SCRN"]["taus"]
    STD_SCRN = std_stats["SCRN"]["taus"]
    QOI_SPG = average_stats["SPG"]["taus"]
    STD_SPG = std_stats["SPG"]["taus"]
    QOI_ESPG = average_stats["SPG Entropy"]["taus"]
    STD_ESPG = std_stats["SPG Entropy"]["taus"]
    QOI_TSESPG = average_stats["Two stages SPG Entropy"]["taus"]
    STD_TSESPG = std_stats["Two stages SPG Entropy"]["taus"]
    plt.semilogy(np.arange(0, num_episodes, step=test_freq), QOI_SCRN, label=r"SCRN, $\alpha = 1$")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_SCRN-STD_SCRN, QOI_SCRN+STD_SCRN, alpha=0.2)
    plt.semilogy(np.arange(0, num_episodes, step=test_freq), QOI_SPG, label=r"SPG, $\alpha = 1$")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_SPG-STD_SPG, QOI_SPG+STD_SPG, alpha=0.2)
    plt.semilogy(np.arange(0, num_episodes, step=test_freq), QOI_ESPG, label=r"ESPG, $\alpha=2$")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_ESPG-STD_ESPG, QOI_ESPG+STD_ESPG, alpha=0.2)
    plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_TSESPG, label=r"TSESPG, $\alpha = 2$")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_TSESPG-STD_TSESPG, QOI_TSESPG+STD_TSESPG, alpha=0.2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper center', bbox_to_anchor=(0.55, -0.2), fancybox=True, shadow=True, ncol=4, fontsize='large')
    plt.xlabel("Number of episodes", fontsize=20)
    plt.ylabel(r"$\frac{J(\theta^{*}) - J(\theta)}{\vert \vert \nabla J(\theta) \vert \vert^{\alpha}}$", fontsize=20)

    plt.figure()
    QOI_SCRN = average_stats["SCRN"]["rewards"]
    STD_SCRN = std_stats["SCRN"]["rewards"]
    QOI_SPG = average_stats["SPG"]["rewards"]
    STD_SPG = std_stats["SPG"]["rewards"]
    QOI_ESPG = average_stats["SPG Entropy"]["rewards"]
    STD_ESPG = std_stats["SPG Entropy"]["rewards"]
    QOI_TSESPG = average_stats["Two stages SPG Entropy"]["rewards"]
    STD_TSESPG = std_stats["Two stages SPG Entropy"]["rewards"]
    plt.plot(np.arange(0, num_episodes), QOI_SCRN, label="SCRN")
    plt.fill_between(np.arange(0, num_episodes), QOI_SCRN-STD_SCRN, QOI_SCRN+STD_SCRN, alpha=0.2)
    plt.plot(np.arange(0, num_episodes), QOI_SPG, label="SPG")
    plt.fill_between(np.arange(0, num_episodes), QOI_SPG-STD_SPG, QOI_SPG+STD_SPG, alpha=0.2)
    plt.plot(np.arange(0, num_episodes), QOI_ESPG, label="ESPG")
    plt.fill_between(np.arange(0, num_episodes), QOI_ESPG-STD_ESPG, QOI_ESPG+STD_ESPG, alpha=0.2)
    plt.plot(np.arange(0, num_episodes), QOI_TSESPG, label="TSESPG")
    plt.fill_between(np.arange(0, num_episodes), QOI_TSESPG-STD_TSESPG, QOI_TSESPG+STD_TSESPG, alpha=0.2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.55, -0.2), fancybox=True, shadow=True, ncol=4, fontsize='large')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Number of episodes", fontsize=20)
    plt.ylabel("Reward during training", fontsize=20)
    plt.savefig(f"figures/{environment}/rewards.png")

    plt.figure()
    QOI_SCRN = average_stats["SCRN"]["obj_estimates"]
    STD_SCRN = std_stats["SCRN"]["obj_estimates"]
    QOI_SPG = average_stats["SPG"]["obj_estimates"]
    STD_SPG = std_stats["SPG"]["obj_estimates"]
    QOI_ESPG = average_stats["SPG Entropy"]["obj_estimates"]
    STD_ESPG = std_stats["SPG Entropy"]["obj_estimates"]
    QOI_TSESPG = average_stats["Two stages SPG Entropy"]["obj_estimates"]
    STD_TSESPG = std_stats["Two stages SPG Entropy"]["obj_estimates"]
    plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_SCRN, label="SCRN")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_SCRN-STD_SCRN, QOI_SCRN+STD_SCRN, alpha=0.2)
    plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_SPG, label="SPG")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_SPG-STD_SPG, QOI_SPG+STD_SPG, alpha=0.2)
    plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_ESPG, label="ESPG")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_ESPG-STD_ESPG, QOI_ESPG+STD_ESPG, alpha=0.2)
    plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_TSESPG, label="TSESPG")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_TSESPG-STD_TSESPG, QOI_TSESPG+STD_TSESPG, alpha=0.2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.55, -0.2), fancybox=True, shadow=True, ncol=4, fontsize='large')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Number of episodes", fontsize=20)
    plt.ylabel("Objective during training", fontsize=20)

    plt.figure()
    QOI_SCRN = average_stats["SCRN"]["grad_estimates"]
    STD_SCRN = std_stats["SCRN"]["grad_estimates"]
    QOI_SPG = average_stats["SPG"]["grad_estimates"]
    STD_SPG = std_stats["SPG"]["grad_estimates"]
    QOI_ESPG = average_stats["SPG Entropy"]["grad_estimates"]
    STD_ESPG = std_stats["SPG Entropy"]["grad_estimates"]
    QOI_TSESPG = average_stats["Two stages SPG Entropy"]["grad_estimates"]
    STD_TSESPG = std_stats["Two stages SPG Entropy"]["grad_estimates"]
    plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_SCRN, label="SCRN")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_SCRN-STD_SCRN, QOI_SCRN+STD_SCRN, alpha=0.2)
    plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_SPG, label="SPG")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_SPG-STD_SPG, QOI_SPG+STD_SPG, alpha=0.2)
    plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_ESPG, label="ESPG")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_ESPG-STD_ESPG, QOI_ESPG+STD_ESPG, alpha=0.2)
    plt.plot(np.arange(0, num_episodes, step=test_freq), QOI_TSESPG, label="TSESPG")
    plt.fill_between(np.arange(0, num_episodes, step=test_freq), QOI_TSESPG-STD_TSESPG, QOI_TSESPG+STD_TSESPG, alpha=0.2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.55, -0.2), fancybox=True, shadow=True, ncol=4, fontsize='large')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Number of episodes", fontsize=20)
    plt.ylabel(r"$\vert \vert \nabla J(\theta) \vert \vert$", fontsize=20)
