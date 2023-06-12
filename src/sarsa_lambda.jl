"""
    SARSALambdaSolver

SARSA-λ implementation for tabular MDPs, assign credits using eligibility traces

Parameters:
- `exp_policy::ExplorationPolicy`:
    Exploration policy to select the actions
- `n_episodes::Int64`:
    Number of episodes to train the Q table
    default: `100`
- `max_episode_length::Int64`:
    Maximum number of steps before the episode is ended
    default: `100`
- `learning_rate::Float64`:
    Learning rate
    default: `0.001`
- `lambda::Float64`:
    Exponential decay parameter for the eligibility traces
    default: `0.5`
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
Base.@kwdef mutable struct SARSALambdaSolver{E<:ExplorationPolicy, RNG<:AbstractRNG} <: Solver
   n_episodes::Int64 = 100
   max_episode_length::Int64 = 100
   learning_rate::Float64 = 0.001
   exploration_policy::E
   Q_vals::Union{Nothing, Matrix{Float64}} = nothing 
   eligibility::Union{Nothing, Matrix{Float64}} = nothing
   lambda::Float64 = 0.5
   eval_every::Int64 = 10 
   n_eval_traj::Int64 = 20
   rng::RNG = Random.default_rng()
   verbose::Bool = true
end

function solve(solver::SARSALambdaSolver, mdp::MDP)
    (;rng, exploration_policy) = solver
    Q = if isnothing(solver.Q_vals)
        zeros(length(states(mdp)), length(actions(mdp)))
    else
        solver.Q_vals
    end::Matrix{Float64}

    ecounts = if isnothing(solver.eligibility)
        zeros(length(states(mdp)), length(actions(mdp)))
    else
        solver.eligibility
    end::Matrix{Float64}

    on_policy = ValuePolicy(mdp, Q)
    sim = RolloutSimulator(rng=rng, max_steps=solver.max_episode_length)
    k = 0
    for i = 1:solver.n_episodes
        s = rand(rng, initialstate(mdp))
        a = action(exploration_policy, on_policy, k, s)
        t = 0
        while !isterminal(mdp, s) && t < solver.max_episode_length
            sp, r = @gen(:sp, :r)(mdp, s, a, rng)
            k += 1
            ap = action(exploration_policy, on_policy, k, sp)
            si = stateindex(mdp, s)
            ai = actionindex(mdp, a)
            spi = stateindex(mdp, sp)
            api = actionindex(mdp, ap)
            delta = r + discount(mdp) * Q[spi,api] - Q[si,ai]
            ecounts[si,ai] += 1
            for es in states(mdp)
                for ea in actions(mdp)
                    esi, eai = stateindex(mdp, es), actionindex(mdp, ea)
                    Q[esi,eai] += solver.learning_rate * delta * ecounts[esi,eai]
                    ecounts[esi,eai] *= discount(mdp) * solver.lambda
                end
            end
            s, a = sp, ap
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

POMDPLinter.@POMDP_require solve(solver::SARSALambdaSolver, problem::Union{MDP,POMDP}) begin
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
