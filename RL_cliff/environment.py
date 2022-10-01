import numpy as np
import matplotlib.pyplot as plt


class CliffEnv:
    """
    Tile layout (36=start, 47=goal, 37-46=cliff)
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
        self.verbose = 0

    def reset(self):
        self.state = self.start_pos
        self.reward = 0
        self.number_of_steps = 0
        self.path = []
        self.end = False

    def end(self):
        return self.end

    def encode_action(self, action):
        if action == 0:
            return "u"
        elif action == 1:
            return "d"
        elif action == 2:
            return "l"
        elif action == 3:
            return "r"

    def inverse_encoding(self, action):
        if action == "u":
            return 0
        elif action == "d":
            return 1
        elif action == "l":
            return 2
        elif action == "r":
            return 3

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def available(self):
        if self.state == 36:
            return ["u", "r"]
        elif self.state == 0:
            return ["d", "r"]
        elif self.state % 12 == 0 and self.state != 0 and self.state != 36:
            return ["u", "d", "r"]
        elif 1 <= self.state < 12:
            return ["d", "l", "r"]
        elif self.state in self.cliff_pos:
            return ["u", "l", "r"]
        elif (self.state + 1) % 12 == 0:
            return ["u", "d", "l"]
        else:
            return ["u", "d", "l", "r"]

    def do_action(self, action):
        action = self.encode_action(action)
        assert action in ["u", "d", "l", "r"]

        if action not in self.available():
            raise ValueError("Chosen infeasible action from the current state...")

        if action == "u":
            self.state -= 12
        if action == "d":
            self.state += 12
        if action == "l":
            self.state -= 1
        if action == "r":
            self.state += 1

        assert 0 <= self.state <= 47

        self.path.append(self.state)

        self.reward -= 0.1
        self.number_of_steps += 1

        if self.state == 47:
            self.end = True
            self.reward += 100
            if self.verbose:
                print(f"===== Goal reached in {self.number_of_steps} steps =====")
        elif self.number_of_steps >= 100:
            self.end = True
        elif self.state in self.cliff_pos:
            self.reward -= 100

        return self.state, self.reward

    def coords(self, state):
        x = int(state % 12)
        y = 3 - int(state / 12)
        return x, y

    def render(self):
        for hor in range(11):
            for ver in range(4):
                plt.plot([hor, hor+1], [ver, ver], color="black", marker="o", linewidth=3, markersize=5)
        for hor in range(12):
            for ver in range(3):
                plt.plot([hor, hor], [ver, ver+1], color="black", marker="o", linewidth=3, markersize=5)

        for j in self.cliff_pos:
            (x, y) = self.coords(j)
            plt.plot(x, y, color="r", marker="X", markersize=10)

        (x, y) = self.coords(self.start_pos)
        plt.plot(x, y, color='green', marker='o', markersize=15)
        (x, y) = self.coords(self.state)
        plt.plot(x, y, color='red', marker="o", markersize=15)
        (x, y) = self.coords(self.goal_pos)
        plt.plot(x, y, color='blue', marker='*', markersize=15)
        # Figure specifications
        plt.axis('off')
        plt.show()


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


def get_state(agent_pos: tuple) -> int:
    """
    Obtain state corresponding to agent position
    """
    x_dim = 12
    (pos_x, pos_y) = agent_pos
    state = x_dim * pos_x + pos_y

    return state


def get_position(state: int) -> tuple:
    """
    Obtain agent position corresponding to state
    """
    x_dim = 12

    pos_x = int(np.floor(state / x_dim))
    pos_y = state % x_dim

    agent_pos = (pos_x, pos_y)

    return agent_pos
