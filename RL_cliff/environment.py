from abc import ABC
from abc import abstractmethod
from collections import namedtuple

import gym
import numpy as np


from gym.spaces import Box
from gym.spaces import Discrete
from gym.utils import seeding
from PIL import Image
from dataclasses import dataclass
from dataclasses import field

VonNeumannMotion = namedtuple('VonNeumannMotion',
                              ['north', 'south', 'west', 'east'],
                              defaults=[[-1, 0], [1, 0], [0, -1], [0, 1]])


@dataclass
class DeepMindColor:
    obstacle = (160, 160, 160)
    free = (224, 224, 224)
    agent = (51, 153, 255)
    goal = (51, 255, 51)
    button = (102, 0, 204)
    interruption = (255, 0, 255)
    box = (0, 102, 102)
    lava = (255, 0, 0)
    water = (0, 0, 255)


color = DeepMindColor


@dataclass
class Object:
    r"""Defines an object with some of its properties.

    An object can be an obstacle, free space or food etc. It can also have properties like impassable, positions.

    """
    name: str
    value: int
    rgb: tuple
    impassable: bool
    positions: list = field(default_factory=list)


class Cliff:
    """
    Tile layout (36=start, 47=goal, 37-46=default_cliff)
    0	1	2	3	4	5	6	7	8	9	10	11
    12	13	14	15	16	17	18	19	20	21	22	23
    24	25	26	27	28	29	30	31	32	33	34	35
    36	37	38	39	40	41	42	43	44	45	46	47
    """
    def __init__(self, cliff_pos=np.arange(37, 47), goal_pos=47):
        self.cliff_pos = cliff_pos
        self.goal_pos = goal_pos
        self.start_pos = 36
        self.state = self.start_pos
        self.end = False
        self.reward = 0
        self.number_of_steps = 0
        self.path = [self.start_pos]

    def reset(self):
        self.state = self.start_pos
        self.reward = 0
        self.number_of_steps = 0
        self.path = []
        self.end = False
        return self.start_pos

    def end(self):
        return self.end

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def step(self, action: int):
        """
        Move agent to new position based on current position and action
        """
        (pos_x, pos_y) = self.state_to_position(self.get_state())

        if action == 0:  # Up
            pos_y = pos_y - 1 if pos_y > 0 else pos_y
        elif action == 1:  # Down
            pos_y = pos_y + 1 if pos_y < 3 else pos_y
        elif action == 2:  # Left
            pos_x = pos_x - 1 if pos_x > 0 else pos_x
        elif action == 3:  # Right
            pos_x = pos_x + 1 if pos_x < 11 else pos_x
        else:  # Infeasible move
            raise Exception("Infeasible move")

        self.state = self.position_to_state((pos_x, pos_y))
        self.number_of_steps += 1

        if self.state == self.goal_pos:
            self.end = True
            self.reward = 100
            print(f"===== Goal reached in {self.number_of_steps} steps =====")
        elif self.number_of_steps >= 100:
            self.end = True
            self.reward = -0.1
        elif self.state in self.cliff_pos:
            self.reward = -100
            self.end = True
            print(f"===== Agent fallen in cliff in position {self.get_state()} =====")
        else:
            self.reward = -0.1

        return self.state, self.reward

    def position_to_state(self, agent_pos: tuple) -> int:
        """
        Obtain state corresponding to agent position
        """
        x_dim = 12
        (pos_x, pos_y) = agent_pos
        state = x_dim * pos_y + pos_x

        return state

    def state_to_position(self, state: int):
        pos_y = int(state / 12)
        pos_x = state % 12

        return pos_x, pos_y

    def get_goal_pos(self):
        return self.goal_pos


class BaseEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 3}
    reward_range = (-float('inf'), float('inf'))

    def __init__(self):
        self.viewer = None
        self.seed()

    @abstractmethod
    def step(self, action):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_image(self):
        pass

    def render(self, mode='rgb array', max_width=500):
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_height, img_width = img.shape[:2]
        ratio = max_width/img_width
        img = Image.fromarray(img).resize([int(ratio*img_width), int(ratio*img_height)])
        img = np.asarray(img)
        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class BaseMaze(ABC):
    def __init__(self, **kwargs):
        objects = self.make_objects()
        assert all([isinstance(obj, Object) for obj in objects])
        self.objects = namedtuple('Objects', map(lambda x: x.name, objects), defaults=objects)()

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def size(self):
        r"""Returns a pair of (height, width). """
        pass

    @abstractmethod
    def make_objects(self):
        r"""Returns a list of defined objects. """
        pass

    def _convert(self, x, name):
        for obj in self.objects:
            pos = np.asarray(obj.positions)
            x[pos[:, 0], pos[:, 1]] = getattr(obj, name, None)
        return x

    def to_name(self):
        x = np.empty(self.size, dtype=object)
        return self._convert(x, 'name')

    def to_value(self):
        x = np.empty(self.size, dtype=int)
        return self._convert(x, 'value')

    def to_rgb(self):
        x = np.empty((*self.size, 3), dtype=np.uint8)
        return self._convert(x, 'rgb')

    def to_impassable(self):
        x = np.empty(self.size, dtype=bool)
        return self._convert(x, 'impassable')

    def __repr__(self):
        return f'{self.__class__.__name__}{self.size}'


class Maze(BaseMaze):
    def __init__(self, **kwargs):
        self.x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                           [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                           [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                           [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        super().__init__(**kwargs)

    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        return free, obstacle, agent, goal


class RandomMaze(BaseEnv):
    def __init__(self):
        super().__init__()

        self.num_steps = 0
        self.maze = Maze()
        self.env_id = 'RandomMaze-v0'
        self.motions = VonNeumannMotion()
        self.x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                           [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                           [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                           [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        self.start_idx = [[1, 1]]
        self.goal_idx = [[9, 9]]

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.end = False
        self.maximum_number_steps = 50

    def end(self):
        return self.end

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        self.num_steps += 1

        if self._is_goal(new_position):
            reward = +1
            self.end = True
        elif not valid:
            reward = -1
            self.end = False
        else:
            if self.num_steps >= self.maximum_number_steps:
                reward = -0.01
                self.end = True
            else:
                reward = -0.01
                self.end = False
        return self.maze.to_value(), reward, self.end, {}, {}

    def get_state(self):
        size_maze = self.maze.size
        (pos_x, pos_y) = self.maze.objects.agent.positions[0]
        state = size_maze[1] * pos_x + pos_y
        return state

    def reset(self):
        self.maze.objects.agent.positions = self.start_idx
        self.maze.objects.goal.positions = self.goal_idx
        self.end = False
        self.num_steps = 0
        return self.maze.to_value()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                print("===Goal is achieved!===")
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


def mark_path(agent: tuple, env: np.array) -> np.array:
    """
    Store path taken by agent
    Only needed for visualization
    """
    (posY, posX) = agent
    env[posY][posX] += 1

    return env


def env_to_text(env: np.array) -> str:
    """
    Convert environment to text format
    Needed for visualization in console
    """
    env = np.where(env >= 1, 1, env)

    env = np.array2string(env, precision=0, separator=" ", suppress_small=False)
    env = env.replace("[[", " |")
    env = env.replace("]]", "|")
    env = env.replace("[", "|")
    env = env.replace("]", "|")
    env = env.replace("1", "x")
    env = env.replace("0", " ")

    return env


def encode_vector(index: int, dim: int) -> list:
    """Encode vector as one-hot vector"""
    vector_encoded = np.zeros((1, dim))
    vector_encoded[0, index] = 1

    return vector_encoded
