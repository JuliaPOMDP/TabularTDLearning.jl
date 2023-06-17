using TabularTDLearning
using POMDPs
using POMDPModels
using POMDPTools
using Random
using Test
import POMDPLinter: @show_requirements

mdp = SimpleGridWorld()
Random.seed!(1)


@testset "qlearning" begin
    rng = MersenneTwister(1)
    solver = QLearningSolver(exploration_policy=EpsGreedyPolicy(mdp, 0.5, rng=rng), learning_rate=0.1, n_episodes=500, max_episode_length=50, eval_every=50, n_eval_traj=100, rng=rng)

    println("Test QLearning requirements: ")
    @show_requirements solve(solver, mdp)

    r = test_solver(solver, mdp, max_steps = 100)
    @test r > 0.0
end

@testset "sarsa" begin
    rng = MersenneTwister(1)
    solver = SARSASolver(exploration_policy=EpsGreedyPolicy(mdp, 0.5, rng=rng), learning_rate=0.1, n_episodes=500, max_episode_length=50, eval_every=50, n_eval_traj=100, rng=rng)

    println("Test SARSA requirements: ")
    @show_requirements solve(solver, mdp)

    r = test_solver(solver, mdp, max_steps = 100)
    @test r > 0.0
end

@testset "sarsa λ" begin
    rng = MersenneTwister(2)
    solver = SARSALambdaSolver(exploration_policy=EpsGreedyPolicy(mdp, 0.5, rng=rng), learning_rate=0.1, lambda=0.9, n_episodes=500, max_episode_length=50, eval_every=50, n_eval_traj=100, rng=rng)

    println("Test SARSALambda requirements: ")
    @show_requirements solve(solver, mdp)

    r = test_solver(solver, mdp, max_steps = 100)
    @test r > 0.0
end

@testset "Prioritized Sweeping" begin
    sgw_mdp = SimpleGridWorld(tprob=1.0)
    solver = PrioritizedSweepingSolver(exploration_policy=EpsGreedyPolicy(sgw_mdp, 0.5), learning_rate=0.1, n_episodes=500, max_episode_length=50, eval_every=50, n_eval_traj=100, pq_threshold=2.0)
    
    println("Test PrioritizedSweeping requirements: ")
    @show_requirements solve(solver, sgw_mdp)

    r = test_solver(solver, sgw_mdp, max_steps = 100)
    @test r > 0.0
end
