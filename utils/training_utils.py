import numpy as np

STATE_DIM = 54
ACTION_DIM = 4


######### UTILITIES FOR PROBABILITIES AND ENTROPY BONUS COMPUTATION #########


def encode_vector(index: int, dim: int) -> list:
    """Encode vector as one-hot vector
    :param index: index corresponding to non-zero entry
    :param dim: dimension of the vector
    :return:
        - a list with the encoded vector
    """
    vector_encoded = np.zeros((1, dim))
    vector_encoded[0, index] = 1
    return list(vector_encoded)


def compute_cum_rewards(gamma: float, t: int, rewards: np.array) -> float:
    """Cumulative reward function
    :param gamma: discount factor for future rewards
    :param t: time
    :param rewards: rewards of the trajectory
    :return
        - The cumulative discounted reward along a trajectory starting from time t up to the end
    """
    cum_reward = 0
    for tau in range(t, len(rewards)):
        cum_reward += gamma ** (tau - t) * rewards[tau]
    return cum_reward


def pi(env, theta) -> np.array:
    """
    Computes the softmax policy, probability distribution of actions in given state, given the parameters' theta
    :param env: environment
    :param theta: policy parameters
    """
    probs = np.zeros(ACTION_DIM)
    for action in range(ACTION_DIM):
        action_encoded = encode_vector(action, ACTION_DIM)
        probs[action] = np.exp(theta[0, env.get_state()].dot(action_encoded[0]))
    return probs / np.sum(probs)


def get_entropy_bonus(action_probs: list) -> float:
    """
    Computes the entropy bonus given a list of action probabilities
    :param action_probs: action probabilities
    :return:
        - Entropy bonus to be added to the reward obtained on a transition
    """
    entropy_bonus = 0
    for prob_action in action_probs:
        entropy_bonus -= prob_action * np.log(prob_action + 1e-5)
    return float(entropy_bonus)


######### UTILITIES FOR OBJECTIVE AND PL CONSTANT COMPUTATION #########


def objective_trajectory(reward_trajectory, gamma):
    """
    Computes the objective function for a given trajectory, thus an unbiased but rough estimate of J(\theta).
    :param reward_trajectory: trajectory of rewards
    :param gamma: discount factor for future rewards
    :return
        - objective function for a given trajectory
    """
    obj = 0
    for t in range(len(reward_trajectory)):
        obj += gamma ** t * reward_trajectory[t]
    return obj


def estimate_objective_and_gradient(env, gamma, theta, entropy_bonus=False, num_episodes=50):
    """
    Off training function to estimate objective and gradient under current policy
    :param env: environment
    :param gamma: discount factor for future rewards
    :param theta: parameter for the policy
    :param entropy_bonus: true if there is entropy regularization in the objective to be estimated
    :param num_episodes: batch size
    :return:
        - estimate of the objective function
        - estimate of the gradient and its norm
    """
    obj = 0
    grad = np.zeros((STATE_DIM, ACTION_DIM))

    for episode in range(num_episodes):

        env.reset()

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
            next_state, reward, _, _, _ = env.step(action)

            state_trajectory.append(state)
            action_trajectory.append(action)
            if entropy_bonus:
                reward += get_entropy_bonus(action_probs)
            reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)

        # Computing objective and grad for the current trajectory
        obj_traj = objective_trajectory(reward_trajectory, gamma)
        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory,
                                                          probs_trajectory, reward_trajectory, gamma)

        if entropy_bonus:
            grad_traj = grad_traj - grad_entropy_bonus(action_trajectory, state_trajectory, reward_trajectory,
                                                       probs_trajectory, gamma)

        obj += obj_traj / num_episodes
        grad += np.reshape(grad_traj, (STATE_DIM, ACTION_DIM)) / num_episodes

    return obj, np.linalg.norm(grad)


######### UTILITIES FOR GRADIENT COMPUTATION #########


def grad_entropy_bonus(action_trajectory, state_trajectory, reward_trajectory, probs_trajectory, gamma):
    """
    Computes the gradient of the entropy bonus term of the objective along a trajectory
    :param action_trajectory: trajectory of actions taken
    :param state_trajectory: state trajectory for an episode
    :param reward_trajectory: reward trajectory for an episode
    :param probs_trajectory: probabilities trajectory for an episode
    :param gamma: discount factor for future rewards
    :return:
        - gradient of the regularized entropy objective
    """
    cum_grad_log_phi_temp = np.zeros((len(reward_trajectory), ACTION_DIM * STATE_DIM))
    cum_grad_log_phi = np.zeros((1, ACTION_DIM * STATE_DIM))
    grad = np.zeros((STATE_DIM, ACTION_DIM))
    grad1 = np.zeros((STATE_DIM, ACTION_DIM))
    grad_collection = grad_log_pi(action_trajectory, probs_trajectory)

    for tau in range(len(reward_trajectory)):
        grad1[state_trajectory[tau], :] += grad_collection[tau]
        grad11 = grad1
        grad11 = np.reshape(grad11, (1, ACTION_DIM * STATE_DIM))

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
    Computes the grad(log(pi(a|s))) for all the pair of (state, action) in the trajectory (without cumulative rewards)
    :param action_trajectory: trajectory of actions
    :param probs_trajectory: trajectory of prob. of policy taking each action
    :return:
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


def grad_trajectory(state_trajectory, action_trajectory, probs_trajectory, reward_trajectory, gamma):
    """
    Computes the grad of objective function for a given trajectory
    :param state_trajectory: trajectory of states
    :param action_trajectory: trajectory of actions
    :param probs_trajectory: trajectory of prob. of policy taking each action
    :param reward_trajectory: rewards of a trajectory
    :param gamma: discount factor for future rewards
    :return:
        - gradient of objective function for a given trajectory
    """
    grad_collection = grad_log_pi(action_trajectory, probs_trajectory)
    grad = np.zeros((STATE_DIM, ACTION_DIM))
    for t in range(len(reward_trajectory)):
        cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)
        grad[state_trajectory[t], :] = grad[state_trajectory[t], :] + cum_reward * grad_collection[t]
    grad = np.reshape(grad, (1, ACTION_DIM * STATE_DIM))
    return grad, grad_collection


######### UTILITIES FOR HESSIAN COMPUTATION #########

def Hessian_log_pi(state_trajectory, action_trajectory, theta):
    """
    Computes the Hessian(log(pi(a|s))) for all the pair of (state, action) in the trajectory
    :param state_trajectory: trajectory of states
    :param action_trajectory: trajectory of actions
    :param theta: parameters policy
    :return:
        - a list of Hessian(log(pi(a|s))) for a given trajectory, i.e.,
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


def Hessian_trajectory(state_trajectory, action_trajectory, reward_trajectory, grad, grad_collection, gamma, theta):
    """
    Computes the grad of objective function for a given trajectory
    :param state_trajectory: states of a trajectory
    :param reward_trajectory: rewards of a trajectory
    :param action_trajectory: actions of a trajectory
    :param grad_collection: a list of grad of log(pi(.|.))
    :param grad: gradient of obj. function
    :param grad_collection: a list of grad of log(pi(.|.))
    :param gamma: discount factor
    :param theta: parameters of policy
    :return:
        - Hessian of objective function for a given trajectory
    """
    grad_log_pi_traj = np.zeros((1, ACTION_DIM * STATE_DIM))
    Hessian_phi = np.zeros((ACTION_DIM * STATE_DIM, ACTION_DIM * STATE_DIM))
    Hessian_log_pi_collect = Hessian_log_pi(state_trajectory, action_trajectory, theta)
    for t in range(len(reward_trajectory)):
        cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)
        s = state_trajectory[t]
        Hessian_phi[s * ACTION_DIM:(s + 1) * ACTION_DIM, s * ACTION_DIM:(s + 1) * ACTION_DIM] \
            += cum_reward * Hessian_log_pi_collect[t]
        grad_log_pi_traj[0, s * ACTION_DIM:(s + 1) * ACTION_DIM] += grad_collection[t]
    Hessian = np.atleast_2d(grad).T @ grad_log_pi_traj + Hessian_phi
    return Hessian


def cubic_subsolver(grad, hessian, L=10, rho=30, eps=1e-3, c_=1, T_eps=10):
    """
    Cubic subsolver for SCRN
    :param grad: estimate of the gradient
    :param hessian: estimate of the Hessian
    :param L: smoothness constant of J(\theta)
    :param rho: Lipschitz constant for Hessian of J(\theta)
    :param eps: accuracy
    :param c_: constant
    :param T_eps: steps of gradient descent on stochastic cubic submodel
    :return:
        - Delta \theta for update of the parameter policy
    """
    g_norm = np.linalg.norm(grad)
    if g_norm > L ** 2 / rho:
        temp = grad @ hessian @ grad.T / rho / g_norm ** 2
        R_c = -temp + np.sqrt(temp ** 2 + 2 * g_norm / rho)
        delta = - R_c * grad / g_norm
    else:
        delta = np.zeros((1, ACTION_DIM * STATE_DIM))
        sigma = c_ * (eps * rho) ** 0.5 / L
        mu = 1.0 / (20.0 * L)
        vec = np.random.normal(0, 1, ACTION_DIM * STATE_DIM)  # *2 + torch.ones(grad.size())
        vec /= np.linalg.norm(vec)
        g_ = grad + sigma * vec
        for _ in range(T_eps):
            delta -= mu * (g_ + delta @ hessian + rho / 2 * np.linalg.norm(delta) * delta)
    return delta
