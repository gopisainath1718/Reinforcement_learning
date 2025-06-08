import gym
from typing import Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from policy import create_blackjack_policy, create_epsilon_policy
from env import *


def generate_episode(env: gym.Env, policy: Callable, es: bool = False, time: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()

    if time:
        for _ in range(459):
            if es and len(episode) == 0:
                action = env.action_space.sample()
            else:
                action = Action(policy(state))
            
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

    else:
        while True:
            if es and len(episode) == 0:
                action = env.action_space.sample()
            else:
                action = policy(state)
            
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

    return episode


def on_policy_mc_evaluation(
    env: gym.Env,
    policy: Callable,
    num_episodes: int,
    gamma: float,
) -> defaultdict:
    """On-policy Monte Carlo policy evaluation. First visits will be used.

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP

    Returns:
        V (defaultdict): The values for each state. V[state] = value.
    """
    # We use defaultdicts here for both V and N for convenience. The states will be the keys.
    V = defaultdict(float)
    N = defaultdict(int)
    returns = defaultdict(float)

    for _ in trange(num_episodes, desc="Episode"):
        episode = generate_episode(env, policy)

        G = 0
        for t in range(len(episode) - 1, -1, -1):
            # TODO Q3a
            # Update V and N here according to first visit MC
            G = gamma*G +  episode[t][2]
            state = episode[t][0]

            for i, value in enumerate(episode):
                if state != value[0]:
                    returns[state] = returns[state] + G
                    N[state] +=1
                if i == t: break

    for key in returns:
        V[key] = returns[key]/N[key]

    return V


def on_policy_mc_control_es(
    env: gym.Env, num_episodes: int, gamma: float
) -> Tuple[defaultdict, Callable]:
    """On-policy Monte Carlo control with exploring starts for Blackjack

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
    """
    # We use defaultdicts here for both Q and N for convenience. The states will be the keys and the values will be numpy arrays with length = num actions
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    # If the state was seen, use the greedy action using Q values.
    # Else, default to the original policy of sticking to 20 or 21.
    policy = create_blackjack_policy(Q)

    for _ in trange(num_episodes, desc="Episode"):
        # TODO Q3b
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        episode = generate_episode(env, policy, es=True)
        sa_pair = [(i[0], i[1])  for i in episode]

        G = 0
        for t in range(len(episode)-1, -1, -1):
             
            G = gamma*G + episode[t][2]

            state = episode[t][0]
            if len(state) == 2:
                state = state[0]
            action = episode[t][1]

            if (state, action) not in sa_pair[:t]:
                N[state][action] +=1
                Q[state][action] += (G - Q[state][action]) / N[state][action]
        
        policy = create_blackjack_policy(Q)


    return Q, policy


def on_policy_mc_control_epsilon_soft(env: gym.Env, num_episodes: int, gamma: float, epsilon: float):


    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    returns = np.zeros(num_episodes)
    for i in trange(num_episodes, desc="Episode"):
        episode = generate_episode(env, policy, time=True)
        G = 0
        visited_states = []
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            print(G)
            pair = (state, action)
            if pair not in visited_states:
                visited_states.append(pair)
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]
        returns[i] = G
    return returns
