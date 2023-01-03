class MDP:
    def __init__(self):
        super().__init__()

        self.num_steps = 0
        self.start_idx = 0
        self.goal_idx = 1
        self.state = self.start_idx
        self.end = False
        self.num_states = 2
        self.num_actions = 2
        self.maximum_number_steps = 20

    def end(self):
        return self.end

    def get_state(self):
        return self.state

    @staticmethod
    def get_optimal_path():
        return [0, 2]

    def get_num_states_actions(self):
        return self.num_states, self.num_actions

    def step(self, action):
        self.num_steps += 1
        if action == 0:
            if self.state == 0:
                reward = 1
            else:
                reward = 2
        else:
            reward = 0
        if self.num_steps >= self.maximum_number_steps:
            self.end = True

        return self.state, reward

    def reset(self):
        self.state = self.start_idx
        self.end = False
        self.num_steps = 0
