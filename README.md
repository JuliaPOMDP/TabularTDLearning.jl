# TabularTDLearning

[![Build Status](https://travis-ci.org/JuliaPOMDP/TabularTDLearning.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/TabularTDLearning.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaPOMDP/TabularTDLearning.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaPOMDP/TabularTDLearning.jl?branch=master)

This repository provides Julia implementations of the following Temporal-Difference algorithms:

- Q-Learning
- SARSA
- SARSA lambda

## Installation

This package relies on [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl). Using POMDPs.jl (should automatically take care of dependencies)

```julia
Pkg.add("POMDPs")
import POMDPs
POMDPs.add("TabularTDLearning")
```

OR (some optional dependencies may be missing)

```julia
Pkg.clone("https://github.com/JuliaPOMDP/TabularTDLearning.jl.git")
```

