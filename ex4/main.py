import algorithms
import gym
import policy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from env import *


def plot_backjack(V, ax1, ax2):

    player = np.arange(12, 21 + 1)
    dealer = np.arange(1, 10 + 1)
    ace = np.array([False, True])
    state_values = np.zeros((len(player), len(dealer), len(ace)))

    for i, player_ in enumerate(player):
        for j, dealer_ in enumerate(dealer):
            for k, ace_ in enumerate(ace):
                state_values[i, j, k] = V[player_, dealer_, ace_]
    
    X, Y = np.meshgrid(dealer, player)

    ax1.plot_wireframe(X, Y, state_values[:, :, 0], color='black')
    ax2.plot_wireframe(X, Y, state_values[:, :, 1], color='black')
 
    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('Player sum')
        ax.set_xlabel('Dealer showing')


def plot_backjack_es(Q, policy, ax1, ax2):

    player = np.arange(12, 21 + 1)
    dealer = np.arange(1, 10 + 1)
    ace = np.array([False, True])
    state_values = np.zeros((len(player), len(dealer), len(ace)))

    for i, player_ in enumerate(player):
        for j, dealer_ in enumerate(dealer):
            for k, ace_ in enumerate(ace):
                action = policy((player_, dealer_, int(ace_)))
                state_values[i, j, k] = Q[player_, dealer_, ace_][action]
    
    X, Y = np.meshgrid(dealer, player)
 
    ax1.plot_wireframe(X, Y, state_values[:, :, 0], color='black')
    ax2.plot_wireframe(X, Y, state_values[:, :, 1], color='black')
 
    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('Player sum')
        ax.set_xlabel('Dealer showing')


def q3a():

    env = gym.make("Blackjack-v1")

    V_10000 = algorithms.on_policy_mc_evaluation(env, policy=policy.default_blackjack_policy, num_episodes=10000, gamma=1)
    V_500000 = algorithms.on_policy_mc_evaluation(env, policy=policy.default_blackjack_policy, num_episodes=500000, gamma=1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), subplot_kw={'projection': '3d'})
    axes[0][0].set_title('No usable ace after 10000 episodes')
    axes[0][1].set_title('Usable ace after 10000 episodes')
    axes[1][0].set_title('No usable ace after 500000 episodes')
    axes[1][1].set_title('Usable ace after 500000 episodes')

    plot_backjack(V_10000,axes[0][0], axes[0][1])
    plot_backjack(V_500000,axes[1][0], axes[1][1])
    plt.show()


def q3b():
    env = gym.make("Blackjack-v1")

    Q, policy = algorithms.on_policy_mc_control_es(env, num_episodes=500000, gamma=1)

    # print(policy((5, 3, True)))
    fig, axes = plt.subplots(ncols=2, figsize=(8,5), subplot_kw={'projection': '3d'})
    axes[0].set_title('No usable ace after 500000 episodes')
    axes[1].set_title('Usable ace after 500000 episodes')

    plot_backjack_es(Q, policy, axes[0], axes[1])

    pi_usable = [[policy((player_, dealer_, True)) for dealer_ in range(1, 10 + 1) ] for player_ in range(12, 21 + 1)]
    pi_non_usable = [[policy((player_, dealer_, False)) for dealer_ in range(1, 10 + 1)] for player_ in range(12, 21 + 1)]


    fig, axes1 = plt.subplots(ncols=2, figsize=(8,5))
    
    axes1[0].set_title('Usable ace after 500000 episodes')
    axes1[1].set_title('No usable ace after 500000 episodes')
    
    axes1[0].set_ylabel('Player sum')
    axes1[0].set_xlabel('Dealer showing')
    axes1[1].set_ylabel('Player sum')
    axes1[1].set_xlabel('Dealer showing')

    axes1[0].imshow(np.fliplr(np.flipud(pi_usable)),extent = [1,10,11,21])
    axes1[1].imshow(np.fliplr(np.flipud(pi_non_usable)),extent = [1,10,11,21])
    plt.show()


def q4a():
    goal_pos = (10, 10)
    env = FourRoomsEnv(goal_pos=goal_pos)
    n_trials = 1
    n_eps = 100
    gamma = 0.99
    # epsilon = 0.1
    agents = [0.1, 0.02, 0]

    trial_returns = []
    trial_optimal_G = []

    for t in range(n_trials):
        returns = []

        for i, epsilon in enumerate(agents):

            r = algorithms.on_policy_mc_control_epsilon_soft(env, n_eps, gamma, epsilon)
            print(r)
            returns.append(r)

        trial_returns.append(returns)
        # trial_optimal_G.append(optimal_G)

    trial_returns = np.array(trial_returns)
    # Transpose the arrays to have shape (agents, trials, steps)
    trial_returns = np.transpose(trial_returns, (1, 0, 2))
    # print(trial_returns.shape)
    print(trial_optimal_G)
    # exit()

    # optimal_G = np.mean(trial_optimal_G)
    # ub_std_error = 1.96 * np.std(trial_optimal_G) / np.sqrt(n_trials)
    # upper_bound = optimal_G * np.ones(len(trial_returns[0, 0]))

    for i, epsilon in enumerate(agents):
        avg_trial_return = np.mean(trial_returns[i], axis=0)
        std_error = 1.96 * np.std(trial_returns[i]) / np.sqrt(n_trials)

        plt.plot(avg_trial_return, label=f"ε={epsilon}")
        plt.fill_between(np.arange(0, len(avg_trial_return)), avg_trial_return - std_error, avg_trial_return + std_error, alpha=0.2)
    
    # plt.plot(upper_bound, label="Upper Bound", linestyle="--")
    # plt.fill_between(np.arange(0, len(avg_trial_return)), upper_bound - ub_std_error, upper_bound + ub_std_error, alpha=0.2)

    plt.xlabel("steps")
    plt.ylabel("average return")
    plt.title(f"average return vs. steps for {n_trials} trials") 
    plt.legend()
    plt.show()


def main():
    # q3a()
    q3b()
    # q4a()
    # q6()


if __name__ == '__main__':
    main()