import numpy as np
import gym

env = gym.make("Taxi-v3")

STATE_DIM = env.observation_space.n
ACTION_DIM = env.action_space.n


def encode_vector(index: int, dim: int) -> list:
    """Encode vector as one-hot vector"""
    vector_encoded = np.zeros((1, dim))
    vector_encoded[0, index] = 1

    return vector_encoded


def pi(state, theta) -> np.array:
    """Policy: probability distribution of actions in given state"""
    probs = np.zeros(ACTION_DIM)
    for action in range(ACTION_DIM):
        action_encoded = encode_vector(action, ACTION_DIM)
        probs[action] = np.exp(theta[0, state].dot(action_encoded[0]))
    return probs / np.sum(probs)


def epsilon_greedy_action(state: int, q_table: np.array, epsilon: float) -> int:
    """
    Select action based on the ε-greedy policy
    Random action with prob. ε, greedy action with prob. 1-ε
    """

    # Random uniform sample from [0,1]
    sample = np.random.random()

    # Set to 'explore' if sample <= ε
    explore = True if sample <= epsilon else False

    if explore:  # Explore
        # Select random action
        action = np.random.choice(4)
    else:  # Exploit:
        # Select action with largest Q-value
        action = np.argmax(q_table[state, :])

    return action


def get_max_qvalue(state: int, q_table: np.array) -> float:
    """Retrieve best Q-value for state from table"""
    maximum_state_value = np.amax(q_table[:, state])
    return maximum_state_value


def compute_cum_rewards(gamma: float, t: int, rewards: np.array) -> float:
    """Cumulative reward function"""
    cum_reward = 0
    for tau in range(t, len(rewards)):
        cum_reward += gamma ** (tau - t) * rewards[tau]
    return cum_reward


def update_action_probabilities(alpha: float, gamma: float, theta: np.array, state_trajectory: list,
                                action_trajectory: list, reward_trajectory: list, probs_trajectory: list) -> np.array:
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


def get_entropy_bonus(action_probs: list) -> float:
    entropy_bonus = 0
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


def estimate_objective_and_gradient(env, gamma, theta, num_episodes=100):
    """
    Off training function to estimate objective and gradient under current policy
    :param gamma: discount factor for future rewards
    :param env: environment
    :param theta: parameter for the policy
    :param num_episodes: batch size
    :return:
    """
    sample_traj = ()
    obj = []
    grad = []

    for episode in range(num_episodes):

        state, _ = env.reset()

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
            if count == 100:
                done = True

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)  # + entropy_bonus
            probs_trajectory.append(action_probs)

            state = next_state

        if episode == 0:
            sample_traj = (state_trajectory, action_trajectory)

        # Computing objective, grad and Hessian for the current trajectory
        obj_traj = objective_trajectory(reward_trajectory, gamma)
        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory,
                                                          probs_trajectory, reward_trajectory, gamma)

        obj.append(obj_traj)
        grad.append(np.linalg.norm(grad_traj))

    return obj, grad, sample_traj


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