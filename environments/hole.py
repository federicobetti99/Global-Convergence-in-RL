from environments.gym_utils import *
from gym.spaces import Box
from gym.spaces import Discrete


class Hole(BaseMaze):
    def __init__(self, **kwargs):
        self.x = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.start_idx = [[3, 0]]
        self.goal_idx = [[1, 5]]
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

        self.start_idx = [[3, 0]]
        self.goal_idx = [[1, 5]]

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
        for i in range(4):
            self.optimal_actions[i] = 3
        for i in range(4, 7):
            self.optimal_actions[i] = 1
        for i in range(7, 11):
            self.optimal_actions[i] = 2
        for i in range(11, 13):
            self.optimal_actions[i] = 0
        self.optimal_actions[15] = 3
        self.optimal_actions[17] = 2
        for i in range(20, 22):
            self.optimal_actions[i] = 0
        for i in range(22, 24):
            self.optimal_actions[i] = 0
        for i in range(31, 33):
            self.optimal_actions[i] = 0
        for i in range(33, 35):
            self.optimal_actions[i] = 0
        for i in range(35, 39):
            self.optimal_actions[i] = 2
        for i in range(39, 42):
            self.optimal_actions[i] = 3
        for i in range(42, 44):
            self.optimal_actions[i] = 0

    def get_optimal_actions(self):
        return self.optimal_actions

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        within_maze, passable = self._is_valid(new_position)
        truncated = False
        self.num_steps += 1
        if self.num_steps >= self.maximum_number_steps:
            self.end = True
            reward = -0.1
        else:
            if within_maze:
                self.maze.objects.agent.positions = [new_position]
                if self._is_goal(new_position):
                    reward = +100
                    self.end = True
                    print(f"==== Goal reached in {self.num_steps} steps ====")
                elif not passable:
                    reward = -100
                    truncated = True
                    self.end = True
                else:
                    reward = -0.1
                    self.end = False
            else:
                self.maze.objects.agent.positions = [current_position]
                reward = -0.1
                self.end = False

        return self.maze.to_value(), reward, self.end, truncated, {}

    def set_state(self, state):
        self.maze.objects.agent.positions = state

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
        return self.maze.to_value(), {}

    def reset_position(self):
        self.maze.objects.agent.positions = self.start_idx
        self.maze.objects.goal.positions = self.goal_idx
        self.num_steps = 0
        self.end = False

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        within_maze = nonnegative and within_edge
        if within_maze:
            passable = not self.maze.to_impassable()[position[0]][position[1]]
        else:
            passable = False
        return within_maze, passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()
