from environments.gym_utils import *
from gym.spaces import Box
from gym.spaces import Discrete


class Hole(BaseMaze):
    def __init__(self, **kwargs):
        self.x = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.start_idx = [[0, 0]]
        self.goal_idx = [[6, 5]]
        super().__init__(**kwargs)

    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, list(np.stack(np.where(self.x == 0), axis=1)))
        obstacle = Object('obstacle', 1, color.obstacle, True, list(np.stack(np.where(self.x == 1), axis=1)))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        return free, obstacle, agent, goal


class RandomHole(BaseEnv):
    def __init__(self):
        super().__init__()

        self.num_steps = 0
        self.maze = Hole()
        self.env_id = 'RandomCliff'
        self.motions = VonNeumannMotion()
        self.x = self.maze.x

        self.start_idx = [[0, 0]]
        self.goal_idx = [[6, 5]]

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.num_states = self.maze.size[0] * self.maze.size[1]
        self.action_space = Discrete(len(self.motions))
        self.end = False
        self.maximum_number_steps = 100

        self.optimal_actions = {k: 0 for k in range(self.num_states)}

    def end(self):
        return self.end

    def get_num_states_actions(self):
        return self.num_states, len(self.motions)

    def compute_optimal_actions(self):
        for i in range(22):
            self.optimal_actions[i] = 1
        for i in range(22, 26):
            self.optimal_actions[i] = 3
        for i in range(26, 29):
            self.optimal_actions[i] = 1
        for i in range(29, 33):
            self.optimal_actions[i] = 2
        for i in range(33, 35):
            self.optimal_actions[i] = 0
        for i in range(37, 40):
            self.optimal_actions[i] = 1
        for i in range(42, 44):
            self.optimal_actions[i] = 0
        for i in range(44, 46):
            self.optimal_actions[i] = 0
        for i in range(48, 51):
            self.optimal_actions[i] = 1
        for i in range(53, 55):
            self.optimal_actions[i] = 0
        for i in range(55, 57):
            self.optimal_actions[i] = 0
        for i in range(59, 62):
            self.optimal_actions[i] = 1
        for i in range(64, 66):
            self.optimal_actions[i] = 0
        for i in range(66, 68):
            self.optimal_actions[i] = 0
        self.optimal_actions[70] = 2
        self.optimal_actions[72] = 3
        for i in range(75, 77):
            self.optimal_actions[i] = 0
        for i in range(77, 79):
            self.optimal_actions[i] = 0
        for i in range(86, 88):
            self.optimal_actions[i] = 0
        for i in range(88, 90):
            self.optimal_actions[i] = 0
        for i in range(90, 94):
            self.optimal_actions[i] = 2
        for i in range(94, 97):
            self.optimal_actions[i] = 3
        for i in range(97, 99):
            self.optimal_actions[i] = 0
        for i in range(99, 121):
            self.optimal_actions[i] = 0

    def get_optimal_actions(self):
        return self.optimal_actions

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        truncated = False
        self.num_steps += 1
        if self.num_steps >= self.maximum_number_steps:
            self.end = True
            reward = -0.1
        else:
            if valid:  # non exiting nor going into the obstacle
                self.maze.objects.agent.positions = [new_position]
                if self._is_goal(new_position):
                    reward = +100
                    self.end = True
                    print(f"==== Goal reached in {self.num_steps} steps ====")
                else:
                    reward = -0.1
            else:
                truncated = True
                reward = -100
                self.end = True

        return self.maze.to_value(), reward, self.end, truncated, {}

    def set_state(self, state):
        self.maze.objects.agent.positions = state

    def get_state(self):
        size_maze = self.maze.size
        (pos_x, pos_y) = self.maze.objects.agent.positions[0]
        state = size_maze[1] * pos_x + pos_y
        return state

    def reset(self):
        free_positions = np.where(self.maze.x == 0)
        free_positions = [item[0] * self.maze.size[0] + item[1] for item in zip(free_positions[0], free_positions[1])]
        random_state = np.random.choice(free_positions)
        random_state = [random_state % self.maze.size[0], int(random_state / self.maze.size[0])]
        self.maze.objects.agent.positions = [random_state]
        self.maze.objects.goal.positions = self.goal_idx
        self.end = False
        self.num_steps = 0
        return self.maze.to_value(), {}

    def reset_position(self, state):
        agent_initial_pos = np.where(state == 2)
        self.maze.objects.agent.positions = [[agent_initial_pos[0], agent_initial_pos[1]]]
        self.num_steps = 0

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        if nonnegative and within_edge:
            passable = not self.maze.to_impassable()[position[0]][position[1]]
            if passable:
                return True
            else:
                return False
        else:
            return False

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()
