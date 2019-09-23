"""
    QLearningSolver

Vanilla Q learning implementation for tabular MDPs

Parameters:
- `exploration_policy::Union{EpsGreedyPolicy, CategoricalTabularPolicy}`:
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
- `rng::AbstractRNG` random number generator
- `verbose::Bool`:
    print information during training
    default: `true`
"""
mutable struct QLearningSolver <: Solver
   n_episodes::Int64
   max_episode_length::Int64
   learning_rate::Float64
   exploration_policy::Union{EpsGreedyPolicy, CategoricalTabularPolicy}
   Q_vals::Union{Nothing, Matrix{Float64}}
   eval_every::Int64
   n_eval_traj::Int64
   rng::AbstractRNG
   verbose::Bool
   function QLearningSolver(exp_policy::Union{EpsGreedyPolicy, CategoricalTabularPolicy};
                            rng=Random.GLOBAL_RNG,
                            n_episodes=100,
                            Q_vals = nothing,
                            max_episode_length=100,
                            learning_rate=0.001,
                            eval_every=10,
                            n_eval_traj=20,
                            verbose=true)
    return new(n_episodes, max_episode_length, learning_rate, exp_policy, Q_vals, eval_every, n_eval_traj, rng, verbose)
    end
end

#TODO add verbose
function solve(solver::QLearningSolver, mdp::MDP)
    rng = solver.rng
    if solver.Q_vals === nothing
        Q = zeros(length(states(mdp)), length(actions(mdp)))
    else
        Q = solver.Q_vals
    end
    exploration_policy = solver.exploration_policy
    sim = RolloutSimulator(rng=rng, max_steps=solver.max_episode_length)

    policy = ValuePolicy(mdp, Q)
    exploration_policy.val = policy

    for i = 1:solver.n_episodes
        s = initialstate(mdp, rng)
        t = 0
        while !isterminal(mdp, s) && t < solver.max_episode_length
            a = action(exploration_policy, s)
            sp, r = gen(DDNOut(:sp, :r), mdp, s, a, rng)
            si = stateindex(mdp, s)
            ai = actionindex(mdp, a)
            spi = stateindex(mdp, sp)
            Q[si, ai] += solver.learning_rate * (r + discount(mdp) * maximum(Q[spi, :]) - Q[si,ai])
            s = sp
            t += 1
        end
        if i % solver.eval_every == 0
            r_tot = 0.0
            for traj in 1:solver.n_eval_traj
                r_tot += simulate(sim, mdp, policy, initialstate(mdp, rng))
            end
            solver.verbose ? println("On Iteration $i, Returns: $(r_tot/solver.n_eval_traj)") : nothing
        end
    end
    return policy
end

@POMDP_require solve(solver::QLearningSolver, problem::Union{MDP,POMDP}) begin
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
