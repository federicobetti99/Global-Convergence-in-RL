from NPG.training_utils import *

STATE_DIM = 2
ACTION_DIM = 2


def policy(env, state, theta) -> np.array:
    """Off policy computation of pi(state)
    :param env: environment
    :param state: state of the agent
    :param theta: policy parameters
    :return:
        - softmax of the learned values
    """
    probs = np.zeros(ACTION_DIM)
    env.set_state(state)
    for action in range(ACTION_DIM):
        action_encoded = encode_vector(action, ACTION_DIM)
        probs[action] = np.exp(theta[0, state].dot(action_encoded[0]))
    return probs / np.sum(probs)


def discrete_SCRN(env, num_episodes=10000, alpha=0.001, gamma=0.8, batch_size=10, SGD=0, entropy_bonus=False,
                  period=1000, test_freq=None) -> (np.array, list):
    """
    Trains a RL agent with SCRN
    :param env: environment (the code works with any environment compatible with the gym syntax)
    :param num_episodes: number of training episodes
    :param alpha: learning rate
    :param gamma: discount factor for future rewards
    :param batch_size: batch size for estimates of gradient and Hessian
    :param SGD: True to simply do SGD update
    :param entropy_bonus: True to regularize the objective with an entropy bonus
    :param period: period for the decay of the learning rate
    :param test_freq: test frequency during training for PL inequality testing
    :return:
        - a dictionary of statistics collected during training
    """
    step_cache = []
    reward_cache = []
    env_cache = []
    name_cache = []

    # Initialize theta and other quantities to collect training statistics
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])
    alpha0 = alpha
    alpha = alpha0
    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)
    tau_estimates = []
    Hessians = np.zeros([num_episodes, STATE_DIM * ACTION_DIM])
    history_probs = np.zeros([num_episodes, STATE_DIM, ACTION_DIM])
    count_reached_goal = np.zeros(num_episodes)
    grad = np.zeros((STATE_DIM * ACTION_DIM))
    Hessian = np.zeros((STATE_DIM * ACTION_DIM, STATE_DIM * ACTION_DIM))
    objective_estimates = []
    gradients_estimates = []
    thetas = []

    # compute optimum J(\theta^*)
    optimum = objective_trajectory(env.get_optimal_path(), gamma)

    # Iterate over episodes
    for episode in range(num_episodes):

        # reset environment
        env.reset()

        if episode >= 1:
            print(episode, ": ", steps_cache[episode - 1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        while not env.end:
            # Get state corresponding to agent position
            state = env.get_state()

            # Get probabilities per action from current policy
            action_probs = pi(env, theta)

            # Select random action according to policy
            action = np.random.choice(ACTION_DIM, p=np.squeeze(action_probs))

            # Move agent to next position
            next_state, reward = env.step(action)

            # collect statistics of current step
            rewards_cache[episode] += reward
            state_trajectory.append(state)
            action_trajectory.append(action)
            if entropy_bonus:
                reward_trajectory.append(reward + get_entropy_bonus(action_probs))
            else:
                reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)
            steps_cache[episode] += 1

        # Computing objective, grad and Hessian for the current trajectory
        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory,
                                                          probs_trajectory, reward_trajectory, gamma)
        Hessian_traj = Hessian_trajectory(state_trajectory, action_trajectory, reward_trajectory, grad_traj,
                                          grad_collection_traj, gamma, theta)

        # adding entropy regularized term to grad
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
            grad = np.zeros((STATE_DIM * ACTION_DIM))
            Hessian = np.zeros((STATE_DIM * ACTION_DIM, STATE_DIM * ACTION_DIM))

        # save solution parameters for the current epoch
        thetas.append(theta)

        if test_freq is not None:
            # test validity of the PL inequality by estimating objective and gradient
            if episode % test_freq == 0:
                estimate_obj, estimate_grad = estimate_objective_and_gradient(env, gamma, theta, entropy_bonus, num_episodes=50)
                tau_estimates.append((optimum - estimate_obj) / estimate_grad)
                objective_estimates.append(estimate_obj)
                gradients_estimates.append(estimate_grad)

    # save caches
    step_cache.append(steps_cache)
    reward_cache.append(rewards_cache)
    env_cache.append(env)

    if SGD == 0:
        name_cache.append("SCRN")
    else:
        name_cache.append("SGD")

    stats = {
        "steps": steps_cache,
        "rewards": rewards_cache,
        "env": env_cache,
        "thetas": thetas,
        "optimum": optimum,
        "taus": tau_estimates,
        "obj_estimates": objective_estimates,
        "grad_estimates": gradients_estimates,
        "history_probs": history_probs,
        "Hessians": Hessians,
        "name": name_cache,
        "goals": count_reached_goal,
    }

    return stats


def discrete_policy_gradient(env, num_episodes=1000, alpha=0.01, gamma=0.8, two_phases_params=None,
                             entropy_bonus=False, period=1000, test_freq=None) -> (np.array, list):
    """
    Trains a RL agent with discrete policy gradient
    :param env: environment
    :param num_episodes: number of training episodes
    :param alpha: learning rate
    :param gamma: discount factor for future rewards
    :param two_phases_params: parameters for the two stages variant of the algorithm
    :param entropy_bonus: True to regularize objective with an entropy bonus
    :param period: period for decay of the learning rate
    :param test_freq: test frequency during training to test validity of PL inequality
    :return:
    """
    step_cache = []
    reward_cache = []
    env_cache = []
    name_cache = []

    alpha0 = alpha
    alpha = alpha0

    # Initialize theta and stats collecting builtins
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    tau_estimates = []
    grad = np.zeros((1, STATE_DIM, ACTION_DIM))
    history_probs = np.zeros([num_episodes, STATE_DIM, ACTION_DIM])
    count_reached_goal = np.zeros(num_episodes)
    objective_estimates = []
    gradients_estimates = []
    thetas = []

    # set parameters for two stages if they are not None
    if two_phases_params is not None:
        initial_batch_size = two_phases_params["B1"]  # initial batch size
        final_batch_size = two_phases_params["B2"]   # final batch size
        change_step = two_phases_params["T"]  # change of step
    else:
        initial_batch_size = 1
        final_batch_size = 1
        change_step = num_episodes

    batch_size = initial_batch_size

    # compute optimum J(\theta^*)
    optimum = objective_trajectory(env.get_optimal_path(), gamma)

    # Iterate over episodes
    for episode in range(num_episodes):

        # reset environment
        env.reset()

        if episode >= 1:
            print(episode, ": ", steps_cache[episode - 1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        while not env.end:
            # Get state corresponding to agent position
            state = env.get_state()

            # Get probabilities per action from current policy
            action_probs = pi(env, theta)

            # Select random action according to policy
            action = np.random.choice(ACTION_DIM, p=np.squeeze(action_probs))

            # Move agent to next position
            next_state, reward = env.step(action)

            # Collect statistics of current step
            rewards_cache[episode] += reward
            state_trajectory.append(state)
            action_trajectory.append(action)
            if entropy_bonus:
                reward_trajectory.append(reward + get_entropy_bonus(action_probs))
            else:
                reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)
            steps_cache[episode] += 1

        if episode % period == 0 and episode > 0:
            alpha = alpha0 / (episode / period)

        if episode > change_step:
            batch_size = final_batch_size

        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory,
                                                          probs_trajectory, reward_trajectory, gamma)

        if entropy_bonus:
            grad_traj = grad_traj - grad_entropy_bonus(action_trajectory, state_trajectory, reward_trajectory,
                                                       probs_trajectory, gamma)
        grad_traj = np.reshape(grad_traj, (1, STATE_DIM, ACTION_DIM))
        grad = grad + grad_traj / batch_size

        # save solution vector after current episode
        thetas.append(theta)

        # Update action probabilities at end of each episode
        if episode % batch_size == 0 and episode > 0:
            grad = np.reshape(grad, (STATE_DIM, ACTION_DIM))
            theta = theta + alpha * grad
            grad = np.zeros((1, STATE_DIM, ACTION_DIM))

        if test_freq is not None:
            # test validity of the PL inequality by estimating objective and gradient
            if episode % test_freq == 0:
                estimate_obj, estimate_grad = estimate_objective_and_gradient(env, gamma, theta, entropy_bonus, num_episodes=50)
                objective_estimates.append(estimate_obj)
                gradients_estimates.append(estimate_grad)
                if entropy_bonus:
                    estimate_grad = estimate_grad ** 2
                tau_estimates.append((optimum - estimate_obj) / estimate_grad)

    # save caches
    step_cache.append(steps_cache)
    reward_cache.append(rewards_cache)
    env_cache.append(env)
    name_cache.append("Discrete policy gradient")

    stats = {
        "steps": steps_cache,
        "rewards": rewards_cache,
        "thetas": thetas,
        "history_probs": history_probs,
        "optimum": optimum,
        "taus": tau_estimates,
        "obj_estimates": objective_estimates,
        "grad_estimates": gradients_estimates,
        "name": name_cache,
        "goals": count_reached_goal
    }

    return stats


def NPG(env, num_episodes=1000, alpha=0.1, gamma=0.8, batch_size=1, period=1000, test_freq=None):
    """
    Natural policy gradient for testing on MDP
    :param env: environment (MDP usually, syntax is compatible only with this one)
    :param num_episodes: number of training episodes
    :param alpha: learning rate
    :param gamma: discount factor for future rewards
    :param batch_size: batch size for estimation of gradient
    :param period: period for learning rate decay
    :param test_freq: test frequency for PL constant experiments
    :return:
    """
    step_cache = []
    reward_cache = []
    env_cache = []
    name_cache = []

    alpha0 = alpha
    alpha = alpha0

    # Initialize theta and stats collecting builtins
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])
    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)
    tau_estimates = []
    grad = np.zeros((1, STATE_DIM, ACTION_DIM))
    Fisher = np.zeros((1, STATE_DIM * ACTION_DIM))
    objective_estimates = []
    gradients_estimates = []

    # compute optimum J(\theta^*)
    optimum = objective_trajectory(env.get_optimal_path(), gamma)

    # Iterate over episodes
    for episode in range(num_episodes):

        # reset environment
        env.reset()

        if episode >= 1:
            print(episode, ": ", steps_cache[episode - 1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        while not env.end:
            # Get state corresponding to agent position
            state = env.get_state()

            # Get probabilities per action from current policy
            action_probs = pi(env, theta)

            # Select random action according to policy
            action = np.random.choice(ACTION_DIM, p=np.squeeze(action_probs))

            # Move agent to next position
            next_state, reward = env.step(action)

            # Save statistics of current step
            rewards_cache[episode] += reward
            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)

            steps_cache[episode] += 1

        # update learning rate according to schedule
        if episode % period == 0 and episode > 0:
            alpha = alpha0 / (episode / period)

        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory, probs_trajectory,
                                                          reward_trajectory, gamma)
        grad_traj = np.reshape(grad_traj, (1, STATE_DIM, ACTION_DIM))
        Fisher_traj = Fisher_matrix_trajectory(state_trajectory, action_trajectory, probs_trajectory)
        grad = grad + grad_traj / batch_size
        Fisher = Fisher + Fisher_traj / batch_size

        # Update action probabilities at end of each episode
        if episode % batch_size == 0 and episode > 0:
            grad = np.reshape(grad, (STATE_DIM * ACTION_DIM))
            theta = np.reshape(theta, (STATE_DIM * ACTION_DIM))
            theta = theta + alpha * np.linalg.pinv(Fisher + 1e-3 * np.eye(Fisher.shape[0])) @ grad  # NPG update
            theta = np.reshape(theta, [1, STATE_DIM, ACTION_DIM])
            grad = np.zeros((1, STATE_DIM, ACTION_DIM))  # reset gradient
            Fisher = np.zeros((1, STATE_DIM * ACTION_DIM))  # reset Fisher matrix

        if test_freq is not None:
            # test validity of PL inequality by estimating objective and gradient
            if episode % test_freq == 0:
                estimate_obj, estimate_grad = estimate_objective_and_gradient(env, gamma, theta, False, num_episodes=50)
                objective_estimates.append(estimate_obj)
                gradients_estimates.append(estimate_grad)
                tau_estimates.append((optimum - estimate_obj) / estimate_grad)

    # save caches
    step_cache.append(steps_cache)
    reward_cache.append(rewards_cache)
    env_cache.append(env)
    name_cache.append("Natural policy gradient")

    stats = {
        "steps": steps_cache,
        "rewards": rewards_cache,
        "theta": theta,
        "optimum": optimum,
        "name": name_cache,
    }

    return stats
