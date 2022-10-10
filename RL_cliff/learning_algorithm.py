import numpy as np

from RL_cliff.actions import (
    compute_cum_rewards
)

from RL_cliff.environment import (
    encode_vector
)

STATE_DIM = 48
ACTION_DIM = 4


def policy(env, state, theta) -> np.array:
    """Off policy computation of pi(state)"""
    probs = np.zeros(ACTION_DIM)
    env.set_state(state)
    for action in range(ACTION_DIM):
        action_encoded = encode_vector(action, ACTION_DIM)
        probs[action] = np.exp(theta[0, state].dot(action_encoded[0]))
    return probs / np.sum(probs)


def pi(env, theta) -> np.array:
    """Policy: probability distribution of actions in given state"""
    probs = np.zeros(ACTION_DIM)
    for action in range(ACTION_DIM):
        action_encoded = encode_vector(action, ACTION_DIM)
        probs[action] = np.exp(theta[0, env.get_state()].dot(action_encoded[0]))
    return probs / np.sum(probs)


def get_entropy_bonus(action_probs: list) -> float:
    entropy_bonus = 0
    # action_probs=action_probs.numpy()
    #  action_probs=np.squeeze(action_probs)
    for prob_action in action_probs:
        entropy_bonus -= prob_action * np.log(prob_action + 1e-5)
    return float(entropy_bonus)


def grad_entropy_bonus(action_trajectory, state_trajectory, reward_trajectory, probs_trajectory, gamma):
    cum_grad_log_phi_temp = np.zeros((len(reward_trajectory), ACTION_DIM * STATE_DIM))
    # cum_grad_log_matrix_temp = np.zeros((STATE_DIM, ACTION_DIM))
    cum_grad_log_phi = np.zeros((1, ACTION_DIM * STATE_DIM))
    grad = np.zeros((STATE_DIM, ACTION_DIM))
    grad1 = np.zeros((STATE_DIM, ACTION_DIM))
    grad11 = np.zeros((STATE_DIM, ACTION_DIM))
    grad_collection = grad_log_pi(action_trajectory, probs_trajectory)

    for tau in range(len(reward_trajectory)):
        grad1[state_trajectory[tau], :] += grad_collection[tau]
        grad11 = np.reshape(grad11, (STATE_DIM, ACTION_DIM))
        grad11 = grad1
        grad11 = np.reshape(grad11, (1, ACTION_DIM * STATE_DIM))
        # cum_grad_log_matrix_temp = np.reshape(cum_grad_log_matrix_temp, (STATE_DIM, ACTION_DIM))
        cum_grad_log_matrix_temp = np.zeros((STATE_DIM, ACTION_DIM))
        for psi in range(tau):
            grad[state_trajectory[psi], :] = grad[state_trajectory[psi], :] + grad_collection[psi]

        cum_grad_log_matrix_temp += grad
        cum_grad_log_matrix_temp = cum_grad_log_matrix_temp * np.log(probs_trajectory[tau] + 1e-5)
        cum_grad_log_phi_temp[tau, :] = np.reshape(cum_grad_log_matrix_temp, (1, ACTION_DIM * STATE_DIM))
        cum_grad_log_phi += gamma ** tau * (grad11[0] + cum_grad_log_phi_temp[tau, :])

    return cum_grad_log_phi


def grad_log_pi(action_trajectory, probs_trajectory):
    """
    This function computes the grad(log(pi(a|s))) for all the pair of (state, action) in the trajectory.
    Inputs:
    - action_trajectory: trajectory of actions
    - probs_trajectory: trajectory of prob. of policy taking each action
    Output:
    - grad_collection: a list of grad(log(pi(a|s))) for a given trajectory
    """

    grad_collection = []
    for t in range(len(action_trajectory)):
        action = action_trajectory[t]
        # Determine action probabilities with policy
        #  action_probs = pi(state)
        action_probs = probs_trajectory[t]

        # Encode action
        phi = encode_vector(action, ACTION_DIM)

        # Construct weighted state-action vector (average phi over all actions)
        weighted_phi = np.zeros((1, ACTION_DIM))

        # For demonstration only, simply copies probability vector
        for action in range(ACTION_DIM):
            action_input = encode_vector(action, ACTION_DIM)
            weighted_phi[0] += action_probs[action] * action_input[0]

        # Return grad (phi - weighted phi)
        grad = phi - weighted_phi

        grad_collection.append(grad[0])
    return grad_collection


def Hessian_log_pi(state_trajectory, action_trajectory, theta):
    """
    This function computes the Hessian(log(pi(a|s))) for all the pair of (state, action) in the trajectory.
    Inputs:
    - state_trajectory: trajectory of states
    - action_trajectory: trajectory of actions
    Outputs:
    - Hessian_collection: a list of Hessian(log(pi(a|s))) for a given trajectory, i.e.,
                          t-th element corresponds to (s_t,a_t)
    """
    Hessian_collection = []
    # computing Hessian_log_pi for a trajectory
    for t in range(len(state_trajectory)):
        # action = action_trajectory[t]
        state = state_trajectory[t]
        Z = np.sum(np.exp(theta[0, state]))
        temp_grad_pi = -np.exp(np.atleast_2d(theta[0, state])).T @ np.ones((1, ACTION_DIM)) / Z ** 2
        for action in range(ACTION_DIM):
            temp_grad_pi[action, action] = (Z * np.exp(theta[0, state, action]) - np.exp(
                theta[0, state, action]) ** 2) / Z ** 2
        Hessian = np.zeros((ACTION_DIM, ACTION_DIM))
        for action in range(ACTION_DIM):
            Hessian = Hessian + np.atleast_2d(temp_grad_pi[:, action]).T @ encode_vector(action, ACTION_DIM)
        Hessian_collection.append(Hessian)
    return Hessian_collection


def objective_trajectory(reward_trajectory, gamma):
    """
    This function computes the objective function for a given trajectory, thus an unbiased estimate of J(\theta).
    Inputs:
    - state_trajectory: trajectory of states
    - action_trajectory: trajectory of actions
    - probs_trajectory: trajectory of prob. of policy taking each action
    - reward_trajectory: rewards of a trajectory
    - gamma: discount factor
    Output:
    - obj: obj. function for a given trajectory
    """
    obj = 0
    for t in range(len(reward_trajectory)):
        obj += gamma ** t * reward_trajectory[t]
    return obj


def grad_trajectory(state_trajectory, action_trajectory, probs_trajectory, reward_trajectory, gamma):
    """
    This function computes the grad of objective function for a given trajectory.
    Inputs: 
    - action_trajectory: trajectory of actions
    - probs_trajectory: trajectory of prob. of policy taking each action
    - reward_trajectory: rewards of a trajectory
    - gamma: discount factor
    Output: 
    - grad: grad of obj. function for a given trajectory
    """
    grad_collection = grad_log_pi(action_trajectory, probs_trajectory)
    grad = np.zeros((STATE_DIM, ACTION_DIM))
    for t in range(len(reward_trajectory)):
        cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)
        grad[state_trajectory[t], :] = grad[state_trajectory[t], :] + cum_reward * grad_collection[t]
    grad = np.reshape(grad, (1, ACTION_DIM * STATE_DIM))
    return grad, grad_collection


def Hessian_trajectory(state_trajectory, action_trajectory, reward_trajectory, grad, grad_collection, gamma, theta):
    """
    This function computes the grad of objective function for a given trajectory.
    Inputs: 
    - state_trajectory: states of a trajectory
    - reward_trajectory: rewards of a trajectory
    - grad_collection: a list of grad of log(pi(.|.))
    - probs_trajectory: prob. of taking each action for states in a trajectory
    - grad: gradient of obj. function
    - grad_collection: a list of grad of log(pi(.|.))
    - gamma: discount factor
    - theta: parameters of policy
    Output: 
    - grad: Hessian of obj. function for a given trajectory
    """

    grad_log_pi_traj = np.zeros((1, ACTION_DIM * STATE_DIM))
    Hessian_phi = np.zeros((ACTION_DIM * STATE_DIM, ACTION_DIM * STATE_DIM))
    Hessian_log_pi_collect = Hessian_log_pi(state_trajectory, action_trajectory, theta)
    for t in range(len(reward_trajectory)):
        cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)
        s = state_trajectory[t]
        Hessian_phi[s * ACTION_DIM:(s + 1) * ACTION_DIM, s * ACTION_DIM:(s + 1) * ACTION_DIM] += cum_reward * \
                                                                                                 Hessian_log_pi_collect[
                                                                                                     t]
        grad_log_pi_traj[0, s * ACTION_DIM:(s + 1) * ACTION_DIM] += grad_collection[t]

    Hessian = np.atleast_2d(grad).T @ grad_log_pi_traj + Hessian_phi  # grad @ np.at-least_2d(grad_log_pi_traj).T +
    # Hessian_phi
    return Hessian


def cubic_subsolver(grad, hessian, l=10, rho=30, eps=1e-3, c_=1, T_eps=10):
    g_norm = np.linalg.norm(grad)
    if g_norm > l ** 2 / rho:
        temp = grad @ hessian @ grad.T / rho / g_norm ** 2
        R_c = -temp + np.sqrt(temp ** 2 + 2 * g_norm / rho)
        delta = - R_c * grad / g_norm
    else:
        delta = np.zeros((1, ACTION_DIM * STATE_DIM))
        sigma = c_ * (eps * rho) ** 0.5 / l
        mu = 1.0 / (20.0 * l)
        vec = np.random.normal(0, 1, ACTION_DIM * STATE_DIM)  # *2 + torch.ones(grad.size())
        vec /= np.linalg.norm(vec)
        g_ = grad + sigma * vec
        for _ in range(T_eps):
            delta -= mu * (g_ + delta @ hessian + rho / 2 * np.linalg.norm(delta) * delta)
    return delta


def discrete_SCRN(env, num_episodes=10000, alpha=0.01, gamma=0.8, batch_size=16, SGD=0, period=1000,
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
    objectives = np.zeros(num_episodes)
    gradients = np.zeros(num_episodes)
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

    # Iterate over episodes
    for episode in range(num_episodes):

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
            action = np.random.choice(4, p=np.squeeze(action_probs))

            # Move agent to next position
            next_state, reward = env.do_action(action)

            # entropy_bonus = get_entropy_bonus(action_probs)
            rewards_cache[episode] += reward  # + entropy_bonus

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)  # + entropy_bonus
            probs_trajectory.append(action_probs)

            steps_cache[episode] += 1

        if next_state == 47:
            # print('state47')
            count_goal_pos += 1
            count_reached_goal[episode] = 1

        # Computing objective, grad and Hessian for the current trajectory
        obj_traj = objective_trajectory(reward_trajectory, gamma)
        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory,
                                                          probs_trajectory, reward_trajectory, gamma)
        Hessian_traj = Hessian_trajectory(state_trajectory, action_trajectory, reward_trajectory, grad_traj,
                                          grad_collection_traj, gamma, theta)
        obj = obj + obj_traj / batch_size
        grad = grad + grad_traj / batch_size
        Hessian = Hessian + Hessian_traj / batch_size

        objectives[episode] = obj_traj
        gradients[episode] = np.linalg.norm(grad_traj)
        Hessians[episode] = np.linalg.eig(Hessian_traj)[0]

        # adding entropy regularized term to grad
        if SGD == 1:
            grad_traj = grad_traj  # - grad_entropy_bonus(action_trajectory, state_trajectory,reward_trajectory,
            # probs_trajectory)
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

        for state in range(48):
            action_probs = policy(env, state, theta)
            history_probs[episode][state, :] = action_probs

    all_probs = np.zeros([STATE_DIM, ACTION_DIM])
    for state in range(48):
        action_probs = policy(env, state, theta)
        all_probs[state] = action_probs

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
        "theta": theta,
        "history_probs": history_probs,
        "objectives": objectives,
        "Hessians": Hessians,
        "gradients": gradients,
        "optimum": optimum,
        "name": name_cache,
        "probs": all_probs,
        "goals": count_reached_goal
    }

    return stats


def discrete_policy_gradient(env, num_episodes=1000, alpha=0.01, gamma=0.8, batch_size=16, SGD=0, period=1000,
                             step_cache=None, reward_cache=None, env_cache=None, name_cache=None) -> (np.array, list):
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

    def get_entropy_bonus(action_probs: list) -> float:
        entropy_bonus = 0
        # action_probs=action_probs.numpy()
        #  action_probs=np.squeeze(action_probs)
        for prob_action in action_probs:
            entropy_bonus -= prob_action * np.log(prob_action + 1e-5)

        return float(entropy_bonus)

    def update_action_probabilities(
            alpha: float,
            gamma: float,
            theta: np.array,
            state_trajectory: list,
            action_trajectory: list,
            reward_trajectory: list,
            probs_trajectory: list,
    ) -> np.array:

        gradients = np.zeros((STATE_DIM, ACTION_DIM))
        obj = 0

        for t in range(len(reward_trajectory)):
            state = state_trajectory[t]
            action = action_trajectory[t]
            cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)

            # Determine action probabilities with policy
            #  action_probs = pi(state)
            action_probs = probs_trajectory[t]

            # Encode action
            phi = encode_vector(action, ACTION_DIM)

            # Construct weighted state-action vector (average phi over all actions)
            weighted_phi = np.zeros((1, ACTION_DIM))

            # For demonstration only, simply copies probability vector
            for action in range(ACTION_DIM):
                action_input = encode_vector(action, ACTION_DIM)
                weighted_phi[0] += action_probs[action] * action_input[0]

            # Return score function (phi - weighted phi)
            score_function = phi - weighted_phi

            # Update theta (only update for current state, no changes for other states)
            theta[0, state] += alpha * (cum_reward * score_function[0])  # + score_function[0])
            gradients[state, :] = cum_reward * score_function[0]
            obj += gamma ** t * reward_trajectory[t]

        return theta, obj, gradients

    # Initialize theta
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)
    count_goal_pos = np.zeros(1)
    objectives = np.zeros(num_episodes)
    gradients = np.zeros(num_episodes)
    Hessians = np.zeros([num_episodes, STATE_DIM * ACTION_DIM])
    history_probs = np.zeros([num_episodes, STATE_DIM, ACTION_DIM])

    optimal_reward_trajectory = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 100]
    optimum = objective_trajectory(optimal_reward_trajectory, gamma)

    count_reached_goal = np.zeros(num_episodes)

    # Iterate over episodes
    for episode in range(num_episodes):

        env.reset()

        if episode >= 1:
            print(episode, ": ", steps_cache[episode-1])

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
            action = np.random.choice(4, p=np.squeeze(action_probs))

            # Move agent to next position
            next_state, reward = env.do_action(action)

            entropy_bonus = get_entropy_bonus(action_probs)
            rewards_cache[episode] += reward + entropy_bonus

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward + entropy_bonus)
            probs_trajectory.append(action_probs)

            steps_cache[episode] += 1

        if next_state == 47:
            # print('state47')
            count_goal_pos += 1
            count_reached_goal[episode] = 1

        if episode % period == 0 and episode > 0:
            alpha = alpha0 / (episode / period)

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

        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory,
                                                          probs_trajectory, reward_trajectory, gamma)
        Hessian_traj = Hessian_trajectory(state_trajectory, action_trajectory, reward_trajectory, grad_traj,
                                          grad_collection_traj, gamma, theta)

        for state in range(48):
            action_probs = policy(env, state, theta)
            history_probs[episode][state, :] = action_probs

        objectives[episode] = obj
        gradients[episode] = np.linalg.norm(gradient)
        Hessians[episode] = np.linalg.eig(Hessian_traj)[0]

    all_probs = np.zeros([STATE_DIM, ACTION_DIM])
    for state in range(48):
        action_probs = policy(env, state, theta)
        all_probs[state] = action_probs

    step_cache.append(steps_cache)
    reward_cache.append(rewards_cache)

    env_cache.append(env)
    name_cache.append("Discrete policy gradient")

    stats = {
        "steps": steps_cache,
        "rewards": rewards_cache,
        "theta": theta,
        "history_probs": history_probs,
        "objectives": objectives,
        "gradients": gradients,
        "Hessians": Hessians,
        "optimum": optimum,
        "name": name_cache,
        "probs": all_probs,
        "goals": count_reached_goal
    }

    return stats
