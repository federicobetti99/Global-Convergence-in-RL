# Global Convergence in Reinforcement Learning
The validity of a gradient dominance property is often assumed for the convergence study of many reinforcement learning algorithms [1]. The latter condition is satisfied in weak form if the objective function satisfies a Fisher non-degeneracy assumptions together with a boundedness requirement on the compatible function approximation error. While Fisher non degeneracy holds for a large class of policies, such as Gaussian policies [2], under a soft-max policy the Fisher information matrix becomes degenerate when the algorithm gets arbitrarily close to a greedy policy. A non-uniform version of the Polyak-Łojasiewicz has still been shown to hold along the trajectories of policy gradient methods [3], but these algorithms require access to the full gradient of the objective which is often unfeasible to compute; on the other hand, the study in this direction for some much more practical stochastic first order and second order methods is still somehow limited. In this work, we show empirically the validity of a gradient dominance property under a soft-max policy along the trajectories of stochastic policy gradient methods and stochastic second order methods, in different discrete reinforcement learning environments. Using the results obtained, we gain further insight on the faster and more stable convergence of the stochastic cubic regularized Newton method (SCRN) over stochastic policy gradient methods. To go further in understanding the performance of second order methods, we make a first attempt at analyzing the natural policy gradient method and the differences with the cubic regularized Newton's method by comparing the second order information enclosed in the Fisher information and in the Hessian of the objective: in particular, we compare the methods on a simple MDP proposed in [4]. Finally, we draw conclusions about the work in the final section, and we present possible developments of the study presented here.

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

## Reproducibility of the results and usage
To reproduce the obtained results shown in the report, run the script `run.py`
by properly choosing an environment, the number of episodes and the test frequency
of the policy for the validity of the PL inequality testing.
We recall that the results shown in the report are obtained for the following
hyperparameter setting:
```python
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
```python
environment = "umaze"  # any environment in ["cliff", "hole", "random_maze", "umaze"]
```
Moreover, please set additionally
```python
num_episodes = 10000
num_avg = 10
test_freq = 50
```
or the values which you chose if you have run again the experiment in `run.py`.

## Report
The report can be found in _.pdf_ format in the folder `report`.

## References
[1] Saeed Masiha, Saber Salehkaleybar, Niao He, Negar Kiyavash, Patrick Thiran, [Stochastic Second-Order Methods Provably Beat SGD For Gradient-Dominated Functions](https://arxiv.org/abs/2205.12856) <br />
[2] Yanli Liu, Kaiqing Zhang, Tamer Başar, Wotao Yin, [An Improved Analysis of Variance-Reduced Policy Gradient and Natural Policy Gradient Methods](https://arxiv.org/abs/2211.07937) <br />
[3] Jincheng Mei, Chenjun Xiao, Csaba Szepesvari, Dale Schuurmans, [On the Global Convergence Rates of Softmax Policy Gradient Methods](https://arxiv.org/abs/2005.06392) <br />
[4] Sham Kakade, [A Natural Policy Gradient](https://papers.nips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html) <br />

## Authors
- Student: Federico Betti
- Professor: Patrick Thiran
- Supervisor: Saber Salehkaleybar
