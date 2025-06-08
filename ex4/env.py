from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np


def register_env(goal_pos) -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('FourRooms-v0')

    Note: the max_episode_steps option controls the time limit of the environment.
    You can remove the argument to make FourRooms run without a timeout.
    """
    register(id="FourRooms-v0", entry_point="env:FourRoomsEnv", max_episode_steps=459)

 
class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
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


class FourRoomsEnv(Env):
    """Four Rooms gym environment.

    This is a minimal example of how to create a custom gym environment. By conforming to the Gym API, you can use the same `generate_episode()` function for both Blackjack and Four Rooms envs.
    """

    def __init__(self, goal_pos=(10, 10)) -> None:
        self.rows = 11
        self.cols = 11

        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        self.walls = [
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

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self.agent_pos = None

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Tuple[int, int]:
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.agent_pos = self.start_pos

        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 1.0
        else:
            done = False
            reward = 0.0


        # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action).
        # You can reuse your code from ex0

        if action == Action.UP or Action.DOWN:
            actions = [action, Action.LEFT, Action.RIGHT]
            probability = [0.9, 0.05, 0.05]
            action_taken = np.random.choice(actions, p=probability)
        if action == Action.LEFT or Action.RIGHT:
            actions = [action, Action.UP, Action.DOWN]
            probability = [0.9, 0.05, 0.05]
            action_taken = np.random.choice(actions, p=probability)

        # TODO calculate the next position using actions_to_dxdy()
        # You can reuse your code from ex0
        final_action = actions_to_dxdy(action_taken)
        next_pos = (self.agent_pos[0] + final_action[0], self.agent_pos[1] + final_action[1])

        # TODO check if next position is feasible
        # If the next position is a wall or out of bounds, stay at current position

        if next_pos in self.walls:
            # next_pos = self.agent_pos
            self.agent_pos = next_pos


        return self.agent_pos, reward, done, {}, {}
