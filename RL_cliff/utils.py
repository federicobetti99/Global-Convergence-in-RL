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
