import numpy as np
import matplotlib.pyplot as plt
from RL_cliff.actions import epsilon_greedy_action


class Cliff:
    """
    Tile layout (36=start, 47=goal, 37-46=cliff)
    0	1	2	3	4	5	6	7	8	9	10	11
    12	13	14	15	16	17	18	19	20	21	22	23
    24	25	26	27	28	29	30	31	32	33	34	35
    36	37	38	39	40	41	42	43	44	45	46	47
    """
    def __init__(self, cliff_pos=np.hstack(([14, 15, 17, 18], np.arange(26, 31))), goal_pos=16):
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

    def end(self):
        return self.end

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def do_action(self, action: int):
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
