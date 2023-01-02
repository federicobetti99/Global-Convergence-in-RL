from NPG.learning_algorithms import *
from NPG.training_utils import *
from NPG.utils import *
from NPG.simple_MDP import *

env = MDP()
stats = NPG(env, num_episodes=10000, alpha=0.0001, gamma=0.8, batch_size=50)

