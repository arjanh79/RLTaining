import numpy as np

from FrozenLake.v1.Environment import Environment


class Agent:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.e = Environment()

        self.y, self.x = 0, 0  # Go to the start

        self.Q_table = np.zeros((4, 5, 5)) - 1 # (4, 5, 5) up, down, left, right
        self.epsilon = 0.99
        self.gamma = 0.9
        self.alpha = 0.15

    def reset_agent(self):
        self.x = 0
        self.y = 0

    def get_available_actions(self):
        # up, down, left, right
        actions = [1, 1, 1, 1]
        if self.x == 0: actions[0] = 0
        if self.x == 4: actions[1] = 0
        if self.y == 0: actions[2] = 0
        if self.y == 4: actions[3] = 0
        return np.array(actions)

    def get_q_action(self):
        available_actions = np.where(self.get_available_actions() == 1)[0]
        if self.epsilon > self.rng.uniform(0, 1):
            return self.rng.choice(available_actions, 1)
        q_values = np.array([self.Q_table[a, self.x, self.y] for a in available_actions])
        return available_actions[np.argmax(q_values)]

    def do_q_move(self, action):
            if action == 0: self.x -= 1 # up
            if action == 1: self.x += 1 # down
            if action == 2: self.y -= 1 # left
            if action == 3: self.y += 1 # right


    def train_single_game(self):
        self.reset_agent()
        for i in range(50):
            self.update_q_table()
            if self.e.get_value(self.x, self.y) != 0:
                return self.e.get_value(self.x, self.y)
        return -1


    def train(self, num_games=10):
        result = []
        for _ in range(num_games):
            result.append(int(self.train_single_game()))
        self.epsilon *= 0.9
        result = [(i + 1) // 2 for i in result]
        print(f'Win: {(sum(result) / num_games) * 100}%')
        self.get_best_route()


    def update_q_table(self):
        old_x = self.x
        old_y = self.y

        action = self.get_q_action()
        self.do_q_move(action)
        reward = self.e.get_value(self.x, self.y)

        max_Q_next = np.max(self.Q_table[:, self.x, self.y])
        old_value = self.Q_table[action, old_x, old_y]

        self.Q_table[action, old_x, old_y] = old_value + self.alpha * (reward + self.gamma * max_Q_next - old_value)

    def get_best_route(self):
        x, y = 0, 0
        result = [(0, 0)]
        while self.e.get_value(x, y) < 1 and len(result) < 50:
            action = np.argmax(self.Q_table[:, x, y])
            if action == 0: x -= 1
            if action == 1: x += 1
            if action == 2: y -= 1
            if action == 3: y += 1
            if (x, y) in result:
                result.append((x, y))
                self.epsilon = 0.995 # We're running around, do more exploring...
                break
            result.append((x, y))
        print(result)
