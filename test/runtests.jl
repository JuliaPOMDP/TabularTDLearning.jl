using TabularTDLearning
using POMDPs
using POMDPModels
using Test

mdp = GridWorld()



solver = QLearningSolver(mdp, learning_rate=0.1, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
println("Test QLearning requirements: ")
@requirements_info solver mdp

policy = solve(solver, mdp)


solver = SARSASolver(mdp, learning_rate=0.1, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
policy = solve(solver, mdp)


solver = SARSALambdaSolver(mdp, learning_rate=0.1, lambda=0.9, n_episodes=5000, max_episode_length=50, eval_every=50, n_eval_traj=100)
policy = solve(solver, mdp)
