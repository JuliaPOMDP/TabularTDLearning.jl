"""
    QLearningSolver

Vanilla Q learning implementation for tabular MDPs

Parameters:
- `exploration_policy::ExplorationPolicy`:
    Exploration policy to select the actions
- `n_episodes::Int64`:
    Number of episodes to train the Q table
    default: `100`
- `max_episode_length::Int64`:
    Maximum number of steps before the episode is ended
    default: `100`
- `learning_rate::Float64`
    Learning rate
    default: `0.001`
- `eval_every::Int64`:
    Frequency at which to evaluate the trained policy
    default: `10`
- `n_eval_traj::Int64`:
    Number of episodes to evaluate the policy
- `rng::AbstractRNG` random number generator
- `verbose::Bool`:
    print information during training
    default: `true`
"""
Base.@kwdef mutable struct QLearningSolver{E<:ExplorationPolicy,RNG<:AbstractRNG} <: Solver
   n_episodes::Int64 = 100
   max_episode_length::Int64 = 100
   learning_rate::Float64 = 0.001
   exploration_policy::E
   Q_vals::Union{Nothing, Matrix{Float64}} = nothing
   eval_every::Int64 = 10
   n_eval_traj::Int64 = 20
   rng::RNG = Random.default_rng()
   verbose::Bool = true
end

function solve(solver::QLearningSolver, mdp::MDP)
    (;rng,exploration_policy) = solver
    γ = discount(mdp)
    Q = if isnothing(solver.Q_vals)
        zeros(length(states(mdp)), length(actions(mdp)))
    else
        solver.Q_vals
    end::Matrix{Float64}

    sim = RolloutSimulator(rng=rng, max_steps=solver.max_episode_length)

    on_policy = ValuePolicy(mdp, Q)
    k = 0
    for i = 1:solver.n_episodes
        s = rand(rng, initialstate(mdp))
        t = 0
        while !isterminal(mdp, s) && t < solver.max_episode_length
            a = action(exploration_policy, on_policy, k, s)
            k += 1
            sp, r = @gen(:sp, :r)(mdp, s, a, rng)
            si = stateindex(mdp, s)
            ai = actionindex(mdp, a)
            spi = stateindex(mdp, sp)
            Q[si, ai] += solver.learning_rate * (r + γ * maximum(@view(Q[spi, :])) - Q[si,ai])
            s = sp
            t += 1
        end
        if i % solver.eval_every == 0
            r_tot = 0.0
            for traj in 1:solver.n_eval_traj
                r_tot += simulate(sim, mdp, on_policy, rand(rng, initialstate(mdp)))
            end
            solver.verbose && println("On Iteration $i, Returns: $(r_tot/solver.n_eval_traj)")
        end
    end
    return on_policy
end

POMDPLinter.@POMDP_require solve(solver::QLearningSolver, problem::Union{MDP,POMDP}) begin
    P = typeof(problem)
    S = statetype(P)
    A = actiontype(P)
    @req initialstate(::P, ::AbstractRNG)
    @req gen(::DDNOut{(:sp, :r)}, ::P, ::S, ::A, ::AbstractRNG)
    @req stateindex(::P, ::S)
    @req states(::P)
    ss = states(problem)
    @req length(::typeof(ss))
    @req actions(::P)
    as = actions(problem)
    @req length(::typeof(as))
    @req actionindex(::P, ::A)
    @req discount(::P)
end
