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
        self.maximum_number_steps = 100

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
        if action == 0:
            self.state = 0
            self.end = True
            reward = 1
            print("==== Stuck in initial state ====")
        else:
            self.state = 1
            self.end = True
            reward = 2
            print("==== Found high reward ====")

        return self.state, reward

    def reset(self):
        self.state = self.start_idx
        self.end = False
        self.num_steps = 0
