# TabularTDLearning

[![CI](https://github.com/JuliaPOMDP/TabularTDLearning.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/TabularTDLearning.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JuliaPOMDP/TabularTDLearning.jl/branch/master/graph/badge.svg?token=vuJ6Ax5SQj)](https://codecov.io/gh/JuliaPOMDP/TabularTDLearning.jl)

This repository provides Julia implementations of the following Temporal-Difference reinforcement learning algorithms:

- Q-Learning
- SARSA
- SARSA lambda
- Prioritized Sweeping

Note that these solvers are tabular, and will only work with MDPs that have discrete state and action spaces.

## Installation

```julia
Pkg.add("TabularTDLearning")
```

## Example

```julia
using POMDPs
using TabularTDLearning
using POMDPModels
using POMDPTools

mdp = SimpleGridWorld()
# use Q-Learning
exppolicy = EpsGreedyPolicy(mdp, 0.01)
solver = QLearningSolver(exploration_policy=exppolicy, learning_rate=0.1, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
policy = solve(solver, mdp)
# Use SARSA
solver = SARSASolver(exploration_policy=exppolicy, learning_rate=0.1, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
policy = solve(solver, mdp)
# Use SARSA lambda
solver = SARSALambdaSolver(exploration_policy=exppolicy, learning_rate=0.1, lambda=0.9, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
policy = solve(solver, mdp)
# Use Prioritized Sweeping
mdp_ps = SimpleGridWorld(tprob=1.0)
solver = PrioritizedSweepingSolver(exploration_policy=exppolicy, learning_rate=0.1, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100,pq_threshold=0.5)
policy = solve(solver,mdp_ps)
```

