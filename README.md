# Global Convergence in Reinforcement Learning
A large class of policies for discrete state-actions environments 
in a reinforcement learning setting satisfies the Fisher non-degeneracy
assumptions, such as Gaussian policies.
The latter condition, together with the boundedness of a 
compatible function approximation error, implies that a weak gradient
dominance property is satisfied. For a soft-max policy, 
the Fisher information matrix becomes degenerate when the policy
becomes arbitrarily close to a greedy one, independently of the
optimality of the latter. In this semester project, 
we show empirically the validity of a gradient dominance property
under a soft-max policy along the trajectories of different RL algorithms
and in different discrete environments.
We then use this results to gain further insight on the superior
behaviour of the stochastic cubic regularized Newton method (SCRN)
with respect to first order stochastic policy gradient methods
in our setting. The analysis shows that convergence to the optimal policy
is faster and more robust for SCRN compared to SPG, with a comparable
computational cost per episode under our frame.
In the second to last section, we briefly compare the second order
information enclosed in the Fisher information and in the Hessian
as a first attempt to compare the natural policy gradient method
with the cubic regularized Newton's method in the stochastic setting.

## Repository description
- `environments` - Implementation of the considered discrete environments
- `figures` - Plots shown in the report
- `NPG` - Implementation of natural policy gradient and testing on a simple MDP
- `results` - Dictionaries with results
- `utils` - Implementation of learning algorithms and utilities
- `run.py` - Script to reproduce results
- `report` - Contains the report of the obtained results in _.pdf_ format
- `plots.ipynb` - Notebook for visualization of obtained results
- `requirements.txt` - Requirements _.txt_ file

## Installation
To clone the following repository, please run
```
https://github.com/federicobetti99/Global-Convergence-in-RL.git
```
Requirements for the needed packages are available in requirements.txt. To install the needed packages, please run:
```
pip install -r requirements.txt
```

### Reproducibility of the results and usage
To reproduce the obtained results shown in the report, run the script `run.py`
by properly choosing an environment, the number of episodes and the test frequency
of the policy for the validity of the PL inequality testing.
We recall that the results shown in the report are obtained for the following
hyperparameter setting:
```
num_episodes = 10000
gamma = 0.8
batch_size = 1 # except for two stages algorithm
alpha = 0.001 # for first order methods, alpha = 1e-4 for SCRN
lambda_ = 1e-5  # for entropy regularized objectives
test_freq = 50  # and 50 episodes of testing to get estimates of objective and gradient
```
Be aware that running the script above and terminating the procedure:
1. Takes a very long time, as 10 averages are carried out (approximately 4 hours).
2. Overwrites the dictionaries stored in the `results` folder.

Hence, should you just want to visualize the obtained results without loading new ones,
we suggest to directly utilize the notebook `plots.ipynb`.
For the correct usage of the latter, set in the first cell the desired
environment, by choosing e.g.
```
environment = "umaze"  # any environment in ["cliff", "hole", "random_maze", "umaze"]
```
Moreover, please set additionally
```
num_episodes = 10000
num_avg = 10
test_freq = 50
```
or the values which you chose if you have run again the experiment in `run.py`.

## Report
The report can be found in _.pdf_ format in the folder `report`.

## Authors
- Student: Federico Betti
- Professor: Patrick Thiran
- Supervisor: Saber Salehkaleybar
