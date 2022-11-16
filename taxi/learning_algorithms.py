import time

from taxi.training_utils import *
import gym
import pygame
from collections import defaultdict
from scipy.special import softmax


env = gym.make("Taxi-v3")
STATE_DIM = env.observation_space.n
ACTION_DIM = env.action_space.n


def policy(env, state, theta) -> np.array:
    """Off policy computation of pi(state)"""
    probs = np.zeros(ACTION_DIM)
    env.set_state(state)
    for action in range(ACTION_DIM):
        action_encoded = encode_vector(action, ACTION_DIM)
        probs[action] = np.exp(theta[0, state].dot(action_encoded[0]))
    return probs / np.sum(probs)


def discrete_SCRN(env, num_episodes=10000, alpha=0.001, gamma=0.8, batch_size=1, SGD=0, period=1000, test_freq=50,
                  step_cache=None, reward_cache=None, env_cache=None, name_cache=None) -> (np.array, list):
    """
    SCRN with discrete policy (manual weight updates)
    """
    if step_cache is None:
        step_cache = []
    if reward_cache is None:
        reward_cache = []
    if env_cache is None:
        env_cache = []
    if name_cache is None:
        name_cache = []

    # Initialize theta
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])

    alpha0 = alpha
    alpha = alpha0

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    tau_estimates = []
    Hessians = np.zeros([num_episodes, STATE_DIM * ACTION_DIM])

    history_probs = np.zeros([num_episodes, STATE_DIM, ACTION_DIM])

    optimal_reward_trajectory = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 100]
    optimum = objective_trajectory(optimal_reward_trajectory, gamma)

    count_goal_pos = np.zeros(1)
    count_reached_goal = np.zeros(num_episodes)

    # Initialize grad and Hessian
    obj = 0
    grad = np.zeros((STATE_DIM * ACTION_DIM))
    Hessian = np.zeros((STATE_DIM * ACTION_DIM, STATE_DIM * ACTION_DIM))

    estimates = {"objectives": [], "gradients": [], "sample_traj": []}

    # Iterate over episodes
    for episode in range(num_episodes):

        state, _ = env.reset()

        if episode >= 1:
            print(episode, ": ", steps_cache[episode - 1], ", Reward: ", rewards_cache[episode-1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        done = False

        count = 0

        while not done:
            # Get state corresponding to agent position
            # state = env.get_state()

            # Get probabilities per action from current policy
            action_probs = pi(state, theta)

            # Select random action according to policy
            action = np.random.choice(ACTION_DIM, p=np.squeeze(action_probs))

            # Move agent to next position
            next_state, reward, done, _, _ = env.step(action)

            count += 1

            if count == 500:
                done = True

            rewards_cache[episode] += reward
            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)

            steps_cache[episode] += 1

            state = next_state

        # Computing objective, grad and Hessian for the current trajectory
        obj_traj = objective_trajectory(reward_trajectory, gamma)
        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory,
                                                          probs_trajectory, reward_trajectory, gamma)
        Hessian_traj = Hessian_trajectory(state_trajectory, action_trajectory, reward_trajectory, grad_traj,
                                          grad_collection_traj, gamma, theta)
        obj = obj + obj_traj / batch_size
        grad = grad + grad_traj / batch_size
        Hessian = Hessian + Hessian_traj / batch_size

        # adding entropy regularized term to grad
        if SGD == 1:
            grad_traj = grad_traj  # - grad_entropy_bonus(action_trajectory, state_trajectory,
            # reward_trajectory, probs_trajectory, gamma)
        obj = obj + obj_traj / batch_size
        grad = grad + grad_traj / batch_size
        Hessian = Hessian + Hessian_traj / batch_size
        if episode % period == 0 and episode > 0:
            alpha = alpha0 / (episode / period)

        # Update action probabilities after collecting each batch
        if episode % batch_size == 0 and episode > 0:
            if SGD == 1:
                Delta = alpha * grad
            else:
                Delta = cubic_subsolver(-grad, -Hessian)  # 0.001*grad#
            Delta = np.reshape(Delta, (STATE_DIM, ACTION_DIM))
            theta = theta + Delta
            obj = 0
            grad = np.zeros((STATE_DIM * ACTION_DIM))
            Hessian = np.zeros((STATE_DIM * ACTION_DIM, STATE_DIM * ACTION_DIM))

        if episode % test_freq == 0:
            estimate_obj, estimate_grad, sample_traj = estimate_objective_and_gradient(env, gamma, theta,
                                                                                       num_episodes=50)
            tau_estimates.append((optimum - np.mean(estimate_obj)) / np.mean(estimate_grad))
            estimates["sample_traj"].append(sample_traj)

    all_probs = np.zeros([STATE_DIM, ACTION_DIM])

    step_cache.append(steps_cache)
    reward_cache.append(rewards_cache)
    env_cache.append(env)
    good_policy = False

    if SGD == 0:
        name_cache.append("SCRN")
    else:
        name_cache.append("SGD")

    stats = {
        "steps": steps_cache,
        "rewards": rewards_cache,
        "env": env_cache,
        "theta": theta,
        "taus": tau_estimates,
        "estimates": estimates,
        "history_probs": history_probs,
        "Hessians": Hessians,
        "optimum": optimum,
        "name": name_cache,
        "probs": all_probs,
        "goals": count_reached_goal,
        "good_policy": good_policy,
    }

    return stats


def discrete_policy_gradient(env, num_episodes=1000, alpha=0.01, gamma=0.8, batch_size=16, SGD=0, entropy_bonus=False,
                             period=1000, test_freq=50, step_cache=None, reward_cache=None, env_cache=None,
                             name_cache=None) -> (np.array, list):
    """
    REINFORCE with discrete policy gradient (manual weight updates)
    """
    if step_cache is None:
        step_cache = []
    if reward_cache is None:
        reward_cache = []
    if env_cache is None:
        env_cache = []
    if name_cache is None:
        name_cache = []

    alpha0 = alpha
    alpha = alpha0

    # Initialize theta
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)
    count_goal_pos = np.zeros(1)

    tau_estimates = []
    Hessians = np.zeros([num_episodes, STATE_DIM * ACTION_DIM])

    history_probs = np.zeros([num_episodes, STATE_DIM, ACTION_DIM])

    optimal_reward_trajectory = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 100]
    optimum = objective_trajectory(optimal_reward_trajectory, gamma)

    count_reached_goal = np.zeros(num_episodes)

    # Iterate over episodes
    for episode in range(num_episodes):
        state, _ = env.reset()

        if episode >= 1:
            print(episode, ": ", steps_cache[episode - 1], ", Reward: ", rewards_cache[episode-1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []
        done = False

        while not done:
            action_probs = pi(state, theta)
            action = np.argmax(np.squeeze(action_probs))

            # Move agent to next position
            next_state, reward, done, info, _ = env.step(action)

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)
            steps_cache[episode] += 1

            state = next_state

        if episode % period == 0 and episode > 0:
            alpha = alpha0 / (episode / period)

        print("=== Starting update ===")
        # Update action probabilities at end of each episode
        theta, obj, gradient = update_action_probabilities(
            alpha,
            gamma,
            theta,
            state_trajectory,
            action_trajectory,
            reward_trajectory,
            probs_trajectory,
        )
        print("=== Finishing update ===")

        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory,
                                                          probs_trajectory, reward_trajectory, gamma)
        Hessian_traj = Hessian_trajectory(state_trajectory, action_trajectory, reward_trajectory, grad_traj,
                                          grad_collection_traj, gamma, theta)

        if test_freq is not None:
            if episode % test_freq == 0:
                estimate_obj, estimate_grad, sample_traj = estimate_objective_and_gradient(env, gamma, theta,
                                                                                       num_episodes=50)
                tau_estimates.append((optimum - np.mean(estimate_obj)) / np.mean(estimate_grad))

    all_probs = np.zeros([STATE_DIM, ACTION_DIM])
    step_cache.append(steps_cache)
    reward_cache.append(rewards_cache)

    env_cache.append(env)
    name_cache.append("Discrete policy gradient")

    stats = {
        "steps": steps_cache,
        "rewards": rewards_cache,
        "theta": theta,
        "history_probs": history_probs,
        "taus": tau_estimates,
        "Hessians": Hessians,
        "optimum": optimum,
        "name": name_cache,
        "probs": all_probs,
        "goals": count_reached_goal
    }

    return stats


def q_learning(env, alpha=0.05, gamma=0.99, num_episodes=1000, action_policy="epsilon_greedy", epsilon_exploration=None,
               epsilon_exploration_rule=None, trace_decay=0, initial_q=0):
    """
    Trains an agent using the Q-Learning algorithm, by playing num_episodes until the end
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
        It is usually "epsilon_greedy" or "softmax_"
    :param trace_decay: trace decay factor for eligibility traces
        If 0, q_learning(0) is implemented without any eligibility trace. If a non-zero float is given in input
        the latter represents the trace decay factor and q_learning(lambda) is implemented
    :param epsilon_exploration: parameter of the exploration policy. Used if epsilon_exploration_rule is None.
        If the policy is epsilon-greedy, at each iteration the action with the highest Q-value is taken with
        probability (1-epsilon_exploration).
        If the policy is softmax, epsilon_exploration is the scaling parameter for the softmax operation
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode.
        If epsilon_exploration_rule is not None, at episode number n during training the parameter for the
        exploration policy is epsilon_exploration_rule(n).
    :param initial_q: initialization value of all Q-values
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training
    """
    # Q-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of action values
    Q = np.zeros((env.observation_space.n, env.action_space.n))  # All Q-values are initialized to initial_q
    # Stats of training
    episode_rewards = np.empty(num_episodes)
    episode_lengths = np.empty(num_episodes)

    assert action_policy in ("epsilon_greedy", "softmax_")
    # Set the default value of the exploration parameter
    if epsilon_exploration is None:
        if action_policy == "epsilon_greedy":
            epsilon_exploration = 0.1
        else:
            epsilon_exploration = 2

    if epsilon_exploration_rule is None:
        def epsilon_exploration_rule(n):
            return epsilon_exploration  # if an exploration rule is not given, it is the constant one

    for itr in range(num_episodes):
        length = 0
        # first state outside the loop
        state, _ = env.reset()
        done = False

        while not done:
            # choose action according to the desired policy
            if np.random.uniform(0, 1) < epsilon_exploration_rule(itr+1):
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(Q[state])  # Exploit learned values

            next_state, reward, done, info, _ = env.step(action)  # Move according to the policy
            length += 1

            if not done:
                next_greedy_action = np.argmax(Q[next_state])
                target = reward + gamma * Q[next_state][next_greedy_action]
            else:
                target = reward  # the fictitious Q-value of Q(next_state)[\cdot] is zero

            Q[state, action] += alpha * (target - Q[state, action])

            # Preparing for the next move
            state = next_state

        episode_rewards[itr] = reward  # reward of the current episode
        episode_lengths[itr] = length  # length of the current episode

    # Dictionary of stats
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }

    return Q, stats
