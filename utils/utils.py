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
