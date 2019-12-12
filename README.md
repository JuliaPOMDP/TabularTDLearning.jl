# TabularTDLearning

[![Build Status](https://travis-ci.org/JuliaPOMDP/TabularTDLearning.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/TabularTDLearning.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaPOMDP/TabularTDLearning.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaPOMDP/TabularTDLearning.jl?branch=master)

This repository provides Julia implementations of the following Temporal-Difference reinforcement learning algorithms:

- Q-Learning
- SARSA
- SARSA lambda

Note that these solvers are tabular, and will only work with MDPs that have discrete state and action spaces.

## Installation

This package relies on [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl). Using POMDPs.jl (should automatically take care of dependencies)

```julia
Pkg.add("POMDPs")
import POMDPs
POMDPs.add("TabularTDLearning")
```

## Example

```julia
using TabularTDLearning
using POMDPPolicies
using POMDPModels

mdp = GridWorld()
# use Q-Learning
exppolicy = EpsGreedyPolicy(mdp, 0.01)
solver = QLearningSolver(exppolicy, learning_rate=0.1, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
policy = solve(solver, mdp)
# Use SARSA
solver = SARSASolver(exppolicy, learning_rate=0.1, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
policy = solve(solver, mdp)
# Use SARSA lambda
solver = SARSALambdaSolver(exppolicy, learning_rate=0.1, lambda=0.9, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
policy = solve(solver, mdp)


```

