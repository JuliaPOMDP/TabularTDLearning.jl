"""
    SARSASolver

SARSA implementation for tabular MDPs.

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
    defaul: `0.001`
- `eval_every::Int64`:
    Frequency at which to evaluate the trained policy
    default: `10`
- `n_eval_traj::Int64`:
    Number of episodes to evaluate the policy
- `rng::AbstractRNG`: random number generator
- `verbose::Bool`:
    print information during training
    default: `true`
"""
@with_kw mutable struct SARSASolver{E<:ExplorationPolicy} <: Solver
   n_episodes::Int64 = 100
   max_episode_length::Int64 = 100
   learning_rate::Float64 = 0.001
   exploration_policy::E 
   Q_vals::Union{Nothing, Matrix{Float64}} = nothing
   eval_every::Int64 = 10
   n_eval_traj::Int64 = 20
   rng::AbstractRNG = Random.GLOBAL_RNG
   verbose::Bool = true
end

function solve(solver::SARSASolver, mdp::Union{MDP,POMDP})
    rng = solver.rng
    if solver.Q_vals === nothing
        Q = zeros(length(states(mdp)), length(actions(mdp)))
    else
        Q = solver.Q_vals
    end
    exploration_policy = solver.exploration_policy
    sim = RolloutSimulator(rng=rng, max_steps=solver.max_episode_length)

    on_policy = ValuePolicy(mdp, Q)
    k = 0 # step counter
    for i = 1:solver.n_episodes
        s = initialstate(mdp, rng)
        a = action(exploration_policy, on_policy, k, s)
        t = 0
        while !isterminal(mdp, s) && t < solver.max_episode_length
            sp, r = gen(DDNOut(:sp, :r), mdp, s, a, rng)
            k += 1
            ap = action(exploration_policy, on_policy, k, sp)
            si = stateindex(mdp, s)
            ai = actionindex(mdp, a)
            spi = stateindex(mdp, sp)
            api = actionindex(mdp, ap)
            Q[si, ai] += solver.learning_rate * (r + discount(mdp) * Q[spi, api] - Q[si,ai])
            s, a = sp, ap
            t += 1
        end
        if i % solver.eval_every == 0
            r_tot = 0.0
            for traj in 1:solver.n_eval_traj
                r_tot += simulate(sim, mdp, on_policy, initialstate(mdp, rng))
            end
            solver.verbose ? println("On Iteration $i, Returns: $(r_tot/solver.n_eval_traj)") : nothing
        end
    end
    return on_policy
end

@POMDP_require solve(solver::SARSASolver, problem::Union{MDP,POMDP}) begin
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
