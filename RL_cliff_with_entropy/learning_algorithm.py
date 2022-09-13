import numpy as np

from environment import (
    init_env,
    mark_path,
    check_game_over,
    encode_vector,
    get_state,
    get_position,
)
from actions import (
    epsilon_greedy_action,
    move_agent,
    get_max_qvalue,
    get_reward,
    compute_cum_rewards
)

import random

STATE_DIM = 48
ACTION_DIM = 4



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
    Output:
    - Hessian_collection: a list of Hessian(log(pi(a|s))) for a given trajectory, i.e., t-th element corresponds to (s_t,a_t)
    """

    Hessian_collection = []
    #computing Hessian_log_pi for a trajectory
    for t in range(len(state_trajectory)):
        action = action_trajectory[t]
        state = state_trajectory[t]
        Z = np.sum(np.exp(theta[0, state]))
        temp_grad_pi = -np.exp(np.atleast_2d(theta[0, state])).T @ np.ones((1, ACTION_DIM))/Z**2
        for action in range(ACTION_DIM):
            temp_grad_pi[action, action] = (Z*np.exp(theta[0, state, action])-np.exp(theta[0, state, action])**2)/Z**2
        Hessian = np.zeros((ACTION_DIM, ACTION_DIM))
        for action in range(ACTION_DIM):
            Hessian = Hessian + np.atleast_2d(temp_grad_pi[:,action]).T @ encode_vector(action, ACTION_DIM)
        Hessian_collection.append(Hessian)
    return Hessian_collection


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
    grad = np.zeros((STATE_DIM,ACTION_DIM))
    for t in range(len(reward_trajectory)):
        cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)
        grad[state_trajectory[t],:] = grad[state_trajectory[t],:] + cum_reward*grad_collection[t]
    grad = np.reshape(grad, (1,ACTION_DIM*STATE_DIM))
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
    
    grad_log_pi_traj = np.zeros((1,ACTION_DIM*STATE_DIM))
    Hessian_phi = np.zeros((ACTION_DIM*STATE_DIM, ACTION_DIM*STATE_DIM))
    Hessian_log_pi_collect = Hessian_log_pi(state_trajectory, action_trajectory, theta)
    for t in range(len(reward_trajectory)):
        cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)
        s = state_trajectory[t]
        Hessian_phi[s*ACTION_DIM:(s+1)*ACTION_DIM, s*ACTION_DIM:(s+1)*ACTION_DIM] =+ cum_reward*Hessian_log_pi_collect[t]
        grad_log_pi_traj[0,s*ACTION_DIM:(s+1)*ACTION_DIM] =+ grad_collection[t]

    Hessian = np.atleast_2d(grad).T @ grad_log_pi_traj + Hessian_phi#grad @ np.atleast_2d(grad_log_pi_traj).T + Hessian_phi
    return Hessian

def cubic_subsolver(grad, hessian, sim_input):
    l = sim_input.l
    rho = sim_input.rho
    eps = sim_input.eps
    c_ = sim_input.c_
    T_eps = sim_input.T_eps
    g_norm = np.linalg.norm(grad)
    # print(g_norm)
    if g_norm > l**2 / rho:
        temp = grad @ hessian @ grad.T / rho / g_norm**2 
        R_c = -temp + np.sqrt(temp**2 + 2 * g_norm / rho)
        delta = -R_c * grad / g_norm
    else:
        delta = np.zeros((1,ACTION_DIM*STATE_DIM))
        sigma = c_ * (eps * rho)**0.5 / l
        mu = 1.0 / (20.0 * l)
        vec = np.random.normal(0, 1, ACTION_DIM*STATE_DIM)#*2 + torch.ones(grad.size())
        vec /= np.linalg.norm(vec)
        g_ = grad + sigma * vec
        # g_ = grad
        for _ in range(T_eps):
            delta -= mu *(g_ + delta @ hessian + rho / 2 * np.linalg.norm(delta) * delta)
    return delta


def discrete_SCRN(sim_input, sim_output) -> (np.array, list):
    """
    SCRN with discrete policy (manual weight updates)
    """

    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    Batch_size = sim_input.batch_size
    alpha0 = sim_input.alpha
    alpha = alpha0

    def softmax(theta: np.array, action_encoded: list, state: int) -> np.float:
        """Softmax function"""
        return np.exp(theta[0, state].dot(action_encoded[0]))

    def pi(state: int) -> np.array:
        """Policy: probability distribution of actions in given state"""
        probs = np.zeros(ACTION_DIM)
        for action in range(ACTION_DIM):
            action_encoded = encode_vector(action, ACTION_DIM)
            probs[action] = softmax(theta, action_encoded, state)
        return probs / np.sum(probs)


    def get_entropy_bonus(action_probs: list) -> float:
        entropy_bonus = 0
        # action_probs=action_probs.numpy()
        #  action_probs=np.squeeze(action_probs)
        for prob_action in action_probs:
            entropy_bonus -= prob_action * np.log(prob_action + 1e-5)

        return float(entropy_bonus)
    
    def grad_entropy_bonus(action_trajectory, state_trajectory,reward_trajectory, probs_trajectory):
        cum_grad_log_phi_temp=np.zeros((len(reward_trajectory),ACTION_DIM*STATE_DIM))
        cum_grad_log_matrix_temp=np.zeros((STATE_DIM,ACTION_DIM))
        cum_grad_log_phi=np.zeros((1, ACTION_DIM*STATE_DIM))
        grad = np.zeros((STATE_DIM,ACTION_DIM))
        grad1 = np.zeros((STATE_DIM,ACTION_DIM))
        grad11 = np.zeros((STATE_DIM,ACTION_DIM))
        grad_collection=grad_log_pi(action_trajectory, probs_trajectory)
        
        for tau in range(len(reward_trajectory)):
            grad1[state_trajectory[tau],:] += grad_collection[tau]
            grad11 = np.reshape(grad11, (STATE_DIM,ACTION_DIM))
            grad11=grad1
            grad11 = np.reshape(grad11, (1,ACTION_DIM*STATE_DIM))
            cum_grad_log_matrix_temp=np.reshape(cum_grad_log_matrix_temp,(STATE_DIM,ACTION_DIM))
            cum_grad_log_matrix_temp=np.zeros((STATE_DIM,ACTION_DIM))
            for psi in range(tau):
                grad[state_trajectory[psi],:] = grad[state_trajectory[psi],:] + grad_collection[psi]
            
            cum_grad_log_matrix_temp += grad
            cum_grad_log_matrix_temp=cum_grad_log_matrix_temp*np.log(probs_trajectory[tau]+ 1e-5)
            cum_grad_log_phi_temp[tau,:]= np.reshape(cum_grad_log_matrix_temp,(1,ACTION_DIM*STATE_DIM)) 
            cum_grad_log_phi += gamma ** (tau) * (grad11[0]+ cum_grad_log_phi_temp[tau,:])
            
        return cum_grad_log_phi

    # Initialize theta
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)
    temp_goal=np.zeros(1)
    count_goal_pos=np.zeros(1)

    #Initialize grad and Hessian
    grad = np.zeros((STATE_DIM*ACTION_DIM))
    Hessian = np.zeros((STATE_DIM*ACTION_DIM, STATE_DIM*ACTION_DIM))
    
    # Iterate over episodes
    for episode in range(num_episodes):

        if episode >= 1:
            print(episode, ":", steps_cache[episode - 1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        # Initialize environment and agent position
        agent_pos, env, cliff_pos, goal_pos, game_over = init_env()

        while not game_over:

            # Get state corresponding to agent position
            state = get_state(agent_pos)

            # Get probabilities per action from current policy
            action_probs = pi(state)

            # Select random action according to policy
            action = np.random.choice(4, p=np.squeeze(action_probs))

            # Move agent to next position
            agent_pos = move_agent(agent_pos, action)

            # Mark visited path
            env = mark_path(agent_pos, env)

            # Determine next state
            next_state = get_state(agent_pos)

            # Compute and store reward
            reward = get_reward(next_state, cliff_pos, goal_pos)
            entropy_bonus = get_entropy_bonus(action_probs)
            rewards_cache[episode] += reward #+ entropy_bonus

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward+entropy_bonus)
            probs_trajectory.append(action_probs)

            # Check whether game is over
            game_over = check_game_over(episode,
                next_state, cliff_pos, goal_pos, steps_cache[episode]
            )
           

            steps_cache[episode] += 1
         
        if next_state == 47:
             #print('state47')
             count_goal_pos+=1   
        
        #Computing grad and Hessian for the current trajectory
        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory, probs_trajectory, reward_trajectory, gamma)
        Hessian_traj = Hessian_trajectory(state_trajectory, action_trajectory, reward_trajectory, grad_traj, grad_collection_traj, gamma, theta)
        grad = grad + grad_traj/Batch_size
        Hessian = Hessian + Hessian_traj/Batch_size
        #print(grad)
        #adding entropy regularized term to grad
        if sim_input.SGD==1:
            grad_traj=grad_traj#-grad_entropy_bonus(action_trajectory, state_trajectory,reward_trajectory, probs_trajectory)
        grad = grad + grad_traj/Batch_size
        Hessian = Hessian + Hessian_traj/Batch_size
        if episode%sim_input.period == 0 and episode > 0:
            alpha = alpha0/(episode/sim_input.period)

        # Update action probabilities after collecting each batch
        if episode%Batch_size == 0 and episode>0:
            #print("Batch is collected!")
            if sim_input.SGD == 1:
                Delta = alpha*grad
            else:
                Delta = cubic_subsolver(-grad, -Hessian, sim_input)#0.001*grad#
            Delta = np.reshape(Delta, (STATE_DIM,ACTION_DIM))
            #print(grad)
            theta = theta + Delta
            grad = np.zeros((STATE_DIM*ACTION_DIM))
            Hessian = np.zeros((STATE_DIM*ACTION_DIM, STATE_DIM*ACTION_DIM))

    if float(count_goal_pos/num_episodes) >= 0.9:
        temp_goal=1
    else:
        temp_goal=0
    #print('temp_goal:',temp_goal)        

    all_probs = np.zeros([STATE_DIM, ACTION_DIM])
    for state in range(48):
        action_probs = pi(state)
        all_probs[state] = action_probs

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)
    if sim_input.SGD == 0:
        sim_output.name_cache.append("SCRN")
    else:
        sim_output.name_cache.append("SGD") 

    return all_probs, sim_output, temp_goal

def discrete_policy_gradient(sim_input, sim_output) -> (np.array, list):
    """
    REINFORCE with discrete policy gradient (manual weight updates)
    """

    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha0 = sim_input.alpha
    alpha = alpha0

    def softmax(theta: np.array, action_encoded: list, state: int) -> np.float:
        """Softmax function"""
        return np.exp(theta[0, state].dot(action_encoded[0]))

    def pi(state: int) -> np.array:
        """Policy: probability distribution of actions in given state"""
        probs = np.zeros(ACTION_DIM)
        for action in range(ACTION_DIM):
            action_encoded = encode_vector(action, ACTION_DIM)
            probs[action] = softmax(theta, action_encoded, state)
        return probs / np.sum(probs)


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
            theta[0, state] += alpha * (cum_reward * score_function[0] + score_function[0])
        return theta

    # Initialize theta
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)
    count_goal_pos=np.zeros(1)
    temp_goal=np.zeros(1)

    # Iterate over episodes
    for episode in range(num_episodes):

        if episode >= 1:
            print(episode, ":", steps_cache[episode - 1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        # Initialize environment and agent position
        agent_pos, env, cliff_pos, goal_pos, game_over = init_env()

        while not game_over:

            # Get state corresponding to agent position
            state = get_state(agent_pos)

            # Get probabilities per action from current policy
            action_probs = pi(state)

            # Select random action according to policy
            action = np.random.choice(4, p=np.squeeze(action_probs))

            # Move agent to next position
            agent_pos = move_agent(agent_pos, action)

            # Mark visited path
            env = mark_path(agent_pos, env)

            # Determine next state
            next_state = get_state(agent_pos)

            # Compute and store reward
            reward = get_reward(next_state, cliff_pos, goal_pos)
            entropy_bonus = get_entropy_bonus(action_probs)
            rewards_cache[episode] += reward #+ entropy_bonus

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward+ entropy_bonus)
            probs_trajectory.append(action_probs)

            # Check whether game is over
            game_over = check_game_over(episode,
                next_state, cliff_pos, goal_pos, steps_cache[episode]
            )

            steps_cache[episode] += 1
        
        if next_state == 47:
             #print('state47')
             count_goal_pos+=1   
        if episode%sim_input.period == 0 and episode>0:
            alpha = alpha0/(episode/sim_input.period)

        # Update action probabilities at end of each episode
        theta = update_action_probabilities(
            alpha,
            gamma,
            theta,
            state_trajectory,
            action_trajectory,
            reward_trajectory,
            probs_trajectory,
        )
    if float(count_goal_pos/num_episodes)>=0.7:
       temp_goal=1
    else:
       temp_goal=0
    #print('temp_goal:',temp_goal)   
    all_probs = np.zeros([STATE_DIM, ACTION_DIM])
    for state in range(48):
        action_probs = pi(state)
        all_probs[state] = action_probs

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)
    sim_output.name_cache.append("Discrete policy gradient")

    return all_probs, sim_output, temp_goal


