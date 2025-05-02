
import numpy as np

from FrozenLake.v2.Environment import Environment

class Agent:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.size_x = 10
        self.size_y = 10
        self.e = Environment(self.size_x, self.size_y)

        self.y, self.x = 0, 0  # Go to the start

        self.epsilon = 0.99
        self.gamma = 0.9
        self.alpha = 0.15

    def reset_agent(self):
        self.x = 0
        self.y = 0

    def get_available_actions(self):
        actions = [1, 1, 1, 1]
        if self.x == 0: actions[0] = 0
        if self.x == (self.size_x - 1): actions[1] = 0
        if self.y == 0: actions[2] = 0
        if self.y == (self.size_y - 1): actions[3] = 0
        return np.array(actions)

    def get_random_move(self):
        available_actions = np.where(self.get_available_actions() == 1)[0]
        return self.rng.choice(available_actions, 1)[0]

    def make_move(self):
        move = self.get_random_move()

        if move == 0: self.x -= 1  # up
        if move == 1: self.x += 1  # down
        if move == 2: self.y -= 1  # left
        if move == 3: self.y += 1  # right

        print(self.x, self.y)

if __name__ == '__main__':
    a = Agent()


