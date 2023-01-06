# Global Convergence in Reinforcement Learning
The validity of a gradient dominance property is often assumed for the convergence study of many reinforcement learning algorithms [2]. A usual assumption which guarantees the latter condition in weak form is the non-degeneracy of the Fisher information matrix, together with a boundedness requirement on the transferred compatible function approximation error [1]. Under a soft-max policy, the Fisher information matrix becomes degenerate when the policy gets arbitrarily close to a greedy one; as a consequence, only a non-uniform version of the Polyak-Łojasiewicz has been established in the literature. It was shown in [3] that along the trajectories of policy gradient algorithms in the deterministic case, but these algorithms require access to the full gradient of the objective which is often unfeasible or too expensive to compute; on the other hand, the study in this direction for much more practical stochastic first order and second order methods is still somehow limited. In this work, we show empirically that also in the stochastic setting a gradient dominance property holds along the trajectories of stochastic policy gradient methods and stochastic second order methods. From the obtained results, we gain further insight on the faster and more stable convergence of the stochastic cubic regularized Newton method (SCRN) over first order methods. To go further in understanding the performance of second order methods, we make a first attempt at analyzing the natural policy gradient method [4] and the differences with the cubic regularized Newton's method by comparing the second order information enclosed in the Fisher information and in the Hessian of the expected return.

## Repository description
- `environments` - Implementation of discrete environments used for the experiments
- `figures` - Plots shown in the report
- `NPG` - Implementation of natural policy gradient and testing on a simple MDP
- `utils` - Implementation of learning algorithms and utilities
- `run.py` - Script to reproduce results
- `report` - Contains the report of the obtained results in pdf format
- `plots.ipynb` - Notebook for visualization of obtained results
- `requirements.txt` - Requirements file

## Installation
To clone the following repository, please run
```
https://github.com/federicobetti99/Global-Convergence-in-RL.git
```
Requirements for the needed packages are available in `requirements.txt`. To install the needed packages, please run:
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
alpha = 0.01 # for first order methods, alpha = 1e-4 for SCRN
test_freq = 50  # and 100 episodes of testing to get estimates of objective and gradient
```
Be aware that running the script above and terminating the procedure takes a
very long time, as 10 averages are carried out
(approximately 7-8 hours, depending on the environment).

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
test_freq = 50
```
or the values which you chose if you have run again the experiment in `run.py`.

## Report
The report of the obtained results can be found in pdf format in the folder `report`.

## References
[1] Saeed Masiha, Saber Salehkaleybar, Niao He, Negar Kiyavash, Patrick Thiran, [Stochastic Second-Order Methods Provably Beat SGD For Gradient-Dominated Functions](https://arxiv.org/abs/2205.12856) <br />
[2] Yanli Liu, Kaiqing Zhang, Tamer Başar, Wotao Yin, [An Improved Analysis of Variance-Reduced Policy Gradient and Natural Policy Gradient Methods](https://arxiv.org/abs/2211.07937) <br />
[3] Jincheng Mei, Chenjun Xiao, Csaba Szepesvari, Dale Schuurmans, [On the Global Convergence Rates of Softmax Policy Gradient Methods](https://arxiv.org/abs/2005.06392) <br />
[4] Sham Kakade, [A Natural Policy Gradient](https://papers.nips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html) <br />

## Authors
- Student: Federico Betti
- Professor: Patrick Thiran
- Supervisor: Saber Salehkaleybar
