from env import BanditEnv
from tqdm import trange
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from agent import EpsilonGreedy, UCB


def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = BanditEnv(k=k)
    env.reset()

    means = env.mean_values()

    # TODO

    total_rewards=[]
    for a in range(num_samples):
        rewards = []
        for i in range(k):

            reward = env.step(i)
            rewards.append(reward)
        total_rewards.append(rewards)

    array =np.array(total_rewards)
    array = array.reshape(2000,10)

    plt.violinplot(array, showmeans=True, showmedians=True)

    for i, mean in enumerate(means):
        plt.text(i+0.7, mean+0.1, f'q*({i+1})={mean:.2f}', color='black', fontsize=9)
    plt.axhline(y= 0, color="black", linestyle=":")

    # Set title and labels
    plt.title("10-armed Testbed ")
    plt.xticks(range(1, 11))
    plt.xlabel("Action")
    plt.ylabel("Reward Distribution")
    plt.yticks(range(-6, 7))

    plt.show()


def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(10)
    agents = [EpsilonGreedy(10, 0, 0), EpsilonGreedy(10, 0, 0.01), EpsilonGreedy(10, 0, 0.1)]

    # Loop over trials
    total_reward = []
    optimal_steps = np.zeros([len(agents), steps], dtype=np.int64)

    optimal_values=[]
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        optimal_action = env.optimal_action()
        optimal_values.append(env.optimal_value())
        agent_rewards = []
        for k, agent in enumerate(agents):
            # TODO For each trial, perform specified number of steps for each type of agent
            agent.reset()
            rewards = []
            for i in range(steps):
                action = agent.choose_action()

                if action == optimal_action:
                    optimal_steps[k][i] += 1

                reward = env.step(action)

                rewards.append(reward)

                # print(f"action = {action}, reward = {reward}")

                agent.update(action=action, reward=reward)
            
            agent_rewards.append(rewards)
        # print(agent_rewards)    
        total_reward.append(agent_rewards)
    
    avg_total_reward = np.mean(total_reward, axis = 0)

    avg_optimal_value = np.mean(optimal_values, axis=0)

    #calculating confidence band
    cb_0 = 1.96 *(avg_total_reward[0].std()/np.sqrt(trials))
    cb_1 = 1.96 *(avg_total_reward[1].std()/np.sqrt(trials))
    cb_2 = 1.96 *(avg_total_reward[2].std()/np.sqrt(trials))
    cb_ub = 1.96 *(avg_optimal_value.std()/np.sqrt(trials))

    # making numpy array into single flat list
    flat_list_0 = [item for sublist in avg_total_reward[0].tolist() for item in sublist]
    flat_list_1 = [item for sublist in avg_total_reward[1].tolist() for item in sublist]
    flat_list_2 = [item for sublist in avg_total_reward[2].tolist() for item in sublist]

    #plot
    x = range(steps)
    plt.plot(x, avg_total_reward[0], color = 'green', label="ε-greedy (Q1 = 0, ε = 0)")
    plt.plot(x, avg_total_reward[1], color = 'red', label="ε-greedy (Q1 = 0, ε = 0.01)")
    plt.plot(x, avg_total_reward[2], color = 'blue', label="ε-greedy (Q1 = 0, ε = 0.1)")
    
    plt.fill_between(x, flat_list_0-cb_0, flat_list_0+cb_0, alpha = 0.2)
    plt.fill_between(x, flat_list_1-cb_1, flat_list_1+cb_1, alpha = 0.2)
    plt.fill_between(x, flat_list_2-cb_2, flat_list_2+cb_2, alpha = 0.2)
    plt.fill_between(x, avg_optimal_value-cb_ub, avg_optimal_value+cb_ub, alpha = 0.2)

    plt.axhline(y = avg_optimal_value, linestyle = '-', color = "black", label="upper bound")

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("Average reward for 2000 steps and 2000 trials")
    plt.show()  

    percentage = (optimal_steps/trials)*100

    plt.plot(x, percentage[0], color='green', label="ε-greedy (Q1 = 0, ε = 0)")
    plt.plot(x, percentage[1], color='red', label="ε-greedy (Q1 = 0, ε = 0.01")
    plt.plot(x, percentage[2], color='blue', label="ε-greedy (Q1 = 0, ε = 0.1")
    plt.xlabel("Steps")
    plt.ylabel("% optimal actions")
    plt.legend()
    plt.title("% optimal actions for 2000 steps and 2000 runs")
    plt.show()


def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(10)
    agents = [EpsilonGreedy(10, 0, 0), EpsilonGreedy(10, 5, 0), EpsilonGreedy(10, 0, 0.1), EpsilonGreedy(10, 5, 0.1), UCB(10, 0, 2, 0.01)]

    # Loop over trials
    total_reward = []
    optimal_steps = np.zeros([len(agents), steps], dtype=np.int64)
    optimal_values=[]

    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        optimal_action = env.optimal_action()
        optimal_values.append(env.optimal_value())
        agent_rewards = []
        for k, agent in enumerate(agents):
        # TODO For each trial, perform specified number of steps for each type of agent
            agent.reset()
            rewards = []
            for i in range(steps):
                action = agent.choose_action()

                if action == optimal_action:
                    optimal_steps[k][i] += 1

                reward = env.step(action)

                rewards.append(reward)

                # print(f"action = {action}, reward = {reward}")

                agent.update(action=action, reward=reward)
            
            agent_rewards.append(rewards)
        # print(agent_rewards)    
        total_reward.append(agent_rewards)
    
    avg_total_reward = np.mean(total_reward, axis = 0)

    avg_optimal_value = np.mean(optimal_values, axis=0)

    #calculating confidence band
    cb_0 = 1.96 *(avg_total_reward[0].std()/np.sqrt(trials))
    cb_1 = 1.96 *(avg_total_reward[1].std()/np.sqrt(trials))
    cb_2 = 1.96 *(avg_total_reward[2].std()/np.sqrt(trials))
    cb_3 = 1.96 *(avg_total_reward[3].std()/np.sqrt(trials))
    cb_4 = 1.96 *(avg_total_reward[4].std()/np.sqrt(trials))
    cb_ub = 1.96 *(avg_optimal_value.std()/np.sqrt(trials))

    # making numpy array into single flat list
    flat_list_0 = [item for sublist in avg_total_reward[0].tolist() for item in sublist]
    flat_list_1 = [item for sublist in avg_total_reward[1].tolist() for item in sublist]
    flat_list_2 = [item for sublist in avg_total_reward[2].tolist() for item in sublist]
    flat_list_3 = [item for sublist in avg_total_reward[3].tolist() for item in sublist]
    flat_list_4 = [item for sublist in avg_total_reward[4].tolist() for item in sublist]

    #plot
    x = range(steps)
    plt.plot(x, avg_total_reward[0], color = 'green', label="ε-greedy (Q1 = 0, ε = 0)")
    plt.plot(x, avg_total_reward[1], color = 'red', label="ε-greedy (Q1 = 5, ε = 0)")
    plt.plot(x, avg_total_reward[2], color = 'blue', label="ε-greedy (Q1 = 0, ε = 0.1)")
    plt.plot(x, avg_total_reward[3], color = 'black', label="ε-greedy (Q1 = 5, ε = 0.1)")
    plt.plot(x, avg_total_reward[4], color = 'orange', label="UCB (c = 2)")
    
    plt.fill_between(x, flat_list_0-cb_0, flat_list_0+cb_0, alpha = 0.2)
    plt.fill_between(x, flat_list_1-cb_1, flat_list_1+cb_1, alpha = 0.2)
    plt.fill_between(x, flat_list_2-cb_2, flat_list_2+cb_2, alpha = 0.2)
    plt.fill_between(x, flat_list_3-cb_3, flat_list_3+cb_3, alpha = 0.2)
    plt.fill_between(x, flat_list_4-cb_4, flat_list_4+cb_4, alpha = 0.2)
    plt.fill_between(x, avg_optimal_value-cb_ub, avg_optimal_value+cb_ub, alpha = 0.2)

    plt.axhline(y = avg_optimal_value, linestyle = '-', label = "upper bound")

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("Average reward for 1000 steps and 2000 trials")
    plt.show() 

    percentage = (optimal_steps/trials)*100

    plt.plot(x, percentage[0], color='green', label="ε-greedy (Q1 = 0, ε = 0)")
    plt.plot(x, percentage[1], color='red', label="ε-greedy (Q1 = 5, ε = 0)")
    plt.plot(x, percentage[2], color='blue', label="ε-greedy (Q1 = 0, ε = 0.1)")
    plt.plot(x, percentage[3], color='black', label="ε-greedy (Q1 = 5, ε = 0.1)")
    plt.plot(x, percentage[4], color = 'orange', label="UCB (c = 2)")

    plt.xlabel("Steps")
    plt.ylabel("% optimal values")
    plt.legend()
    plt.title("% optimal values for 1000 steps and 2000 trials")
    plt.show()


def main(a):
    # TODO run code for all questions

    if a == '4':
        q4(10, 2000)
    elif a == '6':
        q6(10, 2000, 2000)
    elif a == '7':
        q7(10, 2000, 1000)
    else: 
        print("only choose between 4, 6, 7")

if __name__ == "__main__":
    a = input("Enter 4 or 6 or 7:\n")
    main(a)
