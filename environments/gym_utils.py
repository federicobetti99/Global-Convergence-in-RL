from abc import ABC
from abc import abstractmethod
from collections import namedtuple

import gym
import numpy as np

from gym.utils import seeding
from PIL import Image
from dataclasses import dataclass
from dataclasses import field

VonNeumannMotion = namedtuple('VonNeumannMotion',
                              ['up', 'down', 'left', 'right'],
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
