import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from enum import IntEnum


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action):
    """
    Helper function to map action to changes in x and y coordinates

    Args:
        action (Action): taken action

    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


def reset():
    """Return agent to start state"""
    return (0, 0)


# Q1
def simulate(state: Tuple[int, int], action: Action):
    """Simulate function for Four Rooms environment

    Implements the transition function p(next_state, reward | state, action).
    The general structure of this function is:
        1. If goal was reached, reset agent to start state
        2. Calculate the action taken from selected action (stochastic transition)
        3. Calculate the next state from the action taken (accounting for boundaries/walls)
        4. Calculate the reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))
        action (Action): selected action from current agent position (must be of type Action defined above)

    Returns:
        next_state (Tuple[int, int]): next agent position
        reward (float): reward for taking action in state
    """
    # Walls are listed for you
    # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
        (-1,-1),
        (-1,0),
        (-1,1),
        (-1,2),
        (-1,3),
        (-1,4),
        (-1,5),
        (-1,6),
        (-1,7),
        (-1,8),
        (-1,9),
        (-1,10),
        (-1,11),
        (0,11),
        (1,11),
        (2,11),
        (3,11),
        (4,11),
        (5,11),
        (6,11),
        (7,11),
        (8,11),
        (9,11),
        (10,11),
        (11,11),
        (11,10),
        (11,9),
        (11,8),
        (11,7),
        (11,6),
        (11,5),
        (11,4),
        (11,3),
        (11,2),
        (11,1),
        (11,0),
        (11,-1),
        (10,-1),
        (9,-1),
        (8,-1),
        (7,-1),
        (6,-1),
        (5,-1),
        (4,-1),
        (3,-1),
        (2,-1),
        (1,-1),
        (0,-1),

    ]

    # TODO check if goal was reached
    goal_state = (10, 10)

    # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action)
    # action_taken = action
    if action == Action.UP or Action.DOWN:
        actions = [action, Action.LEFT, Action.RIGHT]
        probability = [0.9, 0.05, 0.05]
        noisy_action = np.random.choice(actions, p=probability)
    if action == Action.LEFT or Action.RIGHT:
        actions = [action, Action.UP, Action.DOWN]
        probability = [0.9, 0.05, 0.05]
        noisy_action = np.random.choice(actions, p=probability)

    # TODO calculate the next state and reward given state and action_taken
    # You can use actions_to_dxdy() to calculate the next state
    # Check that the next state is within boundaries and is not a wall
    # One possible way to work with boundaries is to add a boundary wall around environment and
    # simply check whether the next state is a wall

    final_action = actions_to_dxdy(noisy_action)

    if state == goal_state:
        next_state = reset()
    else:
        next_state = (state[0] + final_action[0],state[1] + final_action[1])
        if next_state in walls:
            next_state = state
 
    reward = 0
    if next_state == goal_state:
        reward = 1
        

    return next_state, reward


# Q2
def manual_policy(state: Tuple[int, int]):
    """A manual policy that queries user for action and returns that action

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    action = input("Enter w,a,s,d for action: ")
    
    if action == 'w':
        return Action.UP
    elif action == 'a':
        return Action.LEFT
    elif action == 's':
        return Action.DOWN
    elif action == 'd':
        return Action.RIGHT

# Q2
def agent(
    steps: int = 10000,
    trials: int = 10,
    policy=Callable[[Tuple[int, int]], Action],
):
    """
    An agent that provides actions to the environment (actions are determined by policy), and receives
    next_state and reward from the environment

    The general structure of this function is:
        1. Loop over the number of trials
        2. Loop over total number of steps
        3. While t < steps
            - Get action from policy
            - Take a step in the environment using simulate()1718
            - Keep track of the reward
        4. Compute cumulative reward of trial

    Args:
        steps (int): steps
        trials (int): trials
        policy: a function that represents the current policy. Agent follows policy for interacting with environment.
            (e.g. policy=manual_policy, policy=random_policy)

    """
    # TODO you can use the following structure and add to it as needed
    rewards = []

    for t in range(trials):
        state = reset()
        i = 0

        total_rewards = []
        total_reward = 0
        while i < steps:

            # # TODO select action to take
            action = policy(state)

            # TODO take step in environment using simulate()
            # TODO record the reward

            next_state, reward = simulate(action=action, state=state)
            state = next_state

            total_reward = total_reward + reward
            total_rewards.append(total_reward)

            i = i+1
        rewards.append(total_rewards)

    return rewards



# Q3
def random_policy(state: Tuple[int, int]):
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    return np.random.choice([Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT])



# Q4
def worse_policy(state: Tuple[int, int]):
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    return Action.RIGHT


# Q4
def better_policy(state: Tuple[int, int]):
    """A policy that is better than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
        (-1,-1),
        (-1,0),
        (-1,1),
        (-1,2),
        (-1,3),
        (-1,4),
        (-1,5),
        (-1,6),
        (-1,7),
        (-1,8),
        (-1,9),
        (-1,10),
        (-1,11),
        (0,11),
        (1,11),
        (2,11),
        (3,11),
        (4,11),
        (5,11),
        (6,11),
        (7,11),
        (8,11),
        (9,11),
        (10,11),
        (11,11),
        (11,10),
        (11,9),
        (11,8),
        (11,7),
        (11,6),
        (11,5),
        (11,4),
        (11,3),
        (11,2),
        (11,1),
        (11,0),
        (11,-1),
        (10,-1),
        (9,-1),
        (8,-1),
        (7,-1),
        (6,-1),
        (5,-1),
        (4,-1),
        (3,-1),
        (2,-1),
        (1,-1),
        (0,-1),

    ]

    # mapping for the wall detections around the agent and taking actions accordingly
    action_map = {

        #up,down,left,right
        (0, 0, 0, 0)    : np.random.choice([Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]),
        (1, 0, 0, 0)    : np.random.choice([Action.LEFT, Action.RIGHT]),
        (0, 1, 0, 0)    : np.random.choice([Action.LEFT, Action.RIGHT]),
        (0, 0, 1, 0)    : np.random.choice([Action.UP, Action.DOWN]),
        (0, 0, 0, 1)    : np.random.choice([Action.UP, Action.DOWN]),
        (1, 0, 1, 0)    : np.random.choice([Action.DOWN, Action.RIGHT]),
        (1, 0, 0, 1)    : np.random.choice([Action.DOWN, Action.LEFT]),
        (0, 1, 1, 0)    : np.random.choice([Action.UP, Action.RIGHT]),
        (0, 1, 0, 1)    : np.random.choice([Action.UP, Action.LEFT]),
        (1, 1, 0, 0)    : np.random.choice([Action.RIGHT, Action.LEFT]),
        (0, 0, 1, 1)    : np.random.choice([Action.UP, Action.DOWN]),

    }

    # checking for the walls around the agent and taking action
    action = action_map[(state[0], state[1]+1) in walls, (state[0], state[1]-1) in walls, (state[0]-1, state[1]) in walls, (state[0]+1, state[1]) in walls]

    return action

    


def main():
    # TODO run code for Q2~Q4 and plot results
    # You may be able to reuse the agent() function for each question

    # manual policy
    agent(steps=50, trials=2, policy=manual_policy)

    # random policy
    random_rewards = agent(policy = random_policy)
    avg_random_rewards = np.mean(random_rewards, axis=0)

    # plots for random policy
    for  r in random_rewards:
        plt.plot(range(10000), r, ':')
    plt.plot(range(10000),avg_random_rewards, linewidth='3', color='black', label='random policy')
    plt.xlabel("Steps")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()

    for  r in random_rewards:
        plt.plot(range(10000), r, ':')
    plt.plot(range(10000),avg_random_rewards, linewidth='3', color='black', label='random policy')

    # worse policy
    worst_rewards = agent(policy = worse_policy)
    avg_worst_rewards = np.mean(worst_rewards, axis=0)

    # plots for worse policy
    for  r in worst_rewards:
        plt.plot(range(10000), r, ':')
    plt.plot(range(10000),avg_worst_rewards, linewidth='3', color='green', label='worse policy')

    # better policy
    better_rewards = agent(policy = better_policy)
    avg_better_rewards = np.mean(better_rewards, axis=0)

    # plots for better policy
    for  r in better_rewards:
        plt.plot(range(10000), r, ':')
    plt.plot(range(10000),avg_better_rewards, linewidth='3', color='blue', label='better policy')
    plt.xlabel("Steps")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
