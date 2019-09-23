using TabularTDLearning
using POMDPs
using POMDPModels
using POMDPPolicies
using POMDPTesting
using Random
using Test

mdp = SimpleGridWorld()
Random.seed!(1)


@testset "qlearning" begin
    rng = MersenneTwister(1)
    solver = QLearningSolver(EpsGreedyPolicy(mdp, 0.5, rng=rng), learning_rate=0.1, n_episodes=500, max_episode_length=50, eval_every=50, n_eval_traj=100, rng=rng)

    println("Test QLearning requirements: ")
    @requirements_info solver mdp

    r = test_solver(solver, mdp, max_steps = 100)
    @test r > 0.0
end

@testset "sarsa" begin
    rng = MersenneTwister(1)
    solver = SARSASolver(EpsGreedyPolicy(mdp, 0.5, rng=rng), learning_rate=0.1, n_episodes=500, max_episode_length=50, eval_every=50, n_eval_traj=100, rng=rng)

    println("Test SARSA requirements: ")
    @requirements_info solver mdp

    r = test_solver(solver, mdp, max_steps = 100)
    @test r > 0.0
end

@testset "sarsa λ" begin
    rng = MersenneTwister(2)
    solver = SARSALambdaSolver(EpsGreedyPolicy(mdp, 0.5, rng=rng), learning_rate=0.1, lambda=0.9, n_episodes=500, max_episode_length=50, eval_every=50, n_eval_traj=100, rng=rng)

    println("Test SARSALambda requirements: ")
    @requirements_info solver mdp

    r = test_solver(solver, mdp, max_steps = 100)
    @test r > 0.0
end
