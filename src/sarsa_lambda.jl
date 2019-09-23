"""
    SARSALambdaSolver

SARSA-λ implementation for tabular MDPs, assign credits using eligibility traces

Parameters:
- `exp_policy::Policy`:
    Exploration policy to select the actions
- `n_episodes::Int64`:
    Number of episodes to train the Q table
    default: `100`
- `max_episode_length::Int64`:
    Maximum number of steps before the episode is ended
    default: `100`
- `learning_rate::Float64`:
    Learning rate
    defaul: `0.001`
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
mutable struct SARSALambdaSolver <: Solver
   n_episodes::Int64
   max_episode_length::Int64
   learning_rate::Float64
   exploration_policy::Union{EpsGreedyPolicy, CategoricalTabularPolicy}
   Q_vals::Union{Nothing, Matrix{Float64}}
   eligibility::Union{Nothing, Matrix{Float64}}
   lambda::Float64
   eval_every::Int64
   n_eval_traj::Int64
   rng::AbstractRNG
   verbose::Bool
   function SARSALambdaSolver(exp_policy::Union{EpsGreedyPolicy, CategoricalTabularPolicy};
                            rng=Random.GLOBAL_RNG,
                            n_episodes=100,
                            max_episode_length=100,
                            learning_rate=0.001,
                            Q_vals = nothing,
                            eligibility = nothing,
                            lambda=0.5,
                            eval_every=10,
                            n_eval_traj=20,
                            verbose = true)
    return new(n_episodes, max_episode_length, learning_rate,
               exp_policy, Q_vals,
               eligibility, lambda, eval_every, n_eval_traj, rng, verbose)
    end
end

function solve(solver::SARSALambdaSolver, mdp::MDP)
    rng = solver.rng
    if solver.Q_vals === nothing
        Q = zeros(length(states(mdp)), length(actions(mdp)))
    else
        Q = solver.Q_vals
    end
    if solver.eligibility === nothing
        ecounts = similar(Q)
    else
        ecounts = solver.eligibility
    end
    exploration_policy = solver.exploration_policy
    policy = ValuePolicy(mdp, Q)
    exploration_policy.val = policy
    sim = RolloutSimulator(rng=rng, max_steps=solver.max_episode_length)

    for i = 1:solver.n_episodes
        s = initialstate(mdp, rng)
        a = action(exploration_policy, s)
        t = 0
        while !isterminal(mdp, s) && t < solver.max_episode_length
            sp, r = gen(DDNOut(:sp, :r), mdp, s, a, rng)
            ap = action(exploration_policy, sp)
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
                r_tot += simulate(sim, mdp, policy, initialstate(mdp, rng))
            end
            solver.verbose ? println("On Iteration $i, Returns: $(r_tot/solver.n_eval_traj)") : nothing
        end
    end
    return policy
end

@POMDP_require solve(solver::SARSALambdaSolver, problem::Union{MDP,POMDP}) begin
    P = typeof(problem)
    S = statetype(P)
    A = actiontype(P)
    @req initialstate(::P, ::AbstractRNG)
    @req gen(::DDNOut{(:sp, :r)}, ::P, ::S, ::A, ::AbstractRNG)
    @req state_index(::P, ::S)
    @req action_index(::P, ::A)
    @req discount(::P)
end
