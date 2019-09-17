using TabularTDLearning
using POMDPs
using POMDPModels
using POMDPTesting
using Test

mdp = SimpleGridWorld()


@testset "qlearning" begin
    solver = QLearningSolver(mdp, learning_rate=0.1, n_episodes=500, max_episode_length=50, eval_every=50, n_eval_traj=100)

    println("Test QLearning requirements: ")
    @requirements_info solver mdp

    r = test_solver(solver, mdp, max_steps = 100)
    @test r > 0.0
end

@testset "sarsa" begin
    solver = SARSASolver(mdp, learning_rate=0.1, n_episodes=500, max_episode_length=50, eval_every=50, n_eval_traj=100)

    println("Test SARSA requirements: ")
    @requirements_info solver mdp

    r = test_solver(solver, mdp, max_steps = 100)
    @test r > 0.0
end

@testset "sarsa λ" begin
    solver = SARSALambdaSolver(mdp, learning_rate=0.1, lambda=0.9, n_episodes=500, max_episode_length=50, eval_every=50, n_eval_traj=100)

    println("Test SARSALambda requirements: ")
    @requirements_info solver mdp

    r = test_solver(solver, mdp, max_steps = 100)
    @test r > 0.0
end
