from NPG.learning_algorithms import *
from NPG.training_utils import *
from NPG.utils import *
from NPG.simple_MDP import *

env = MDP()
stats_NPG = discrete_policy_gradient(env, num_episodes=100, alpha=0.0001, gamma=0.95)

