import numpy as np


class Environment:
    def __init__(self):
        self.lake = [[0, 0, 0, 0, -1], [0, 0, 0, 0, 0], [0, -1, -1, -1, 0], [0, -1, 0, 0, 0], [0, 0, 0, 0, 1]]
        self.lake = np.array(self.lake)
        self.print_lake()

    def print_lake(self):
        print(self.lake)

    def get_value(self, x, y):
        return self.lake[x, y]


class Agent:
    def __init__(self):
        self.e = Environment()
        self.x = 0
        self.y = 0

    def get_available_actions(self):
        # up, down, left, right
        result = [1, 1, 1, 1]
        if self.x == 0:
            result[0] = 0
        if self.x == 4:
            result[1] = 0
        if self.y == 0:
            result[2] = 0
        if self.y == 4:
            result[3] = 0
        return np.array(result)

    def random_move(self):
        available_actions = np.where(self.get_available_actions() == 1)[0]
        action = np.random.choice(available_actions, 1)
        match action:
            case 0: # up
                self.x -= 1
            case 1: # down
                self.x += 1
            case 2: # left
                self.y -= 1
            case 3: # right
                self.y += 1

    def play_single_game(self):
        for i in range(50):
            self.random_move()
            if self.e.get_value(self.x, self.y) != 0:
                return self.e.get_value(self.x, self.y)
        return -1

    def get_location(self):
        return self.x, self.y

    def reset_agent(self):
        self.x = 0
        self.y = 0

    def play_num_games(self, num_games=10):
        result = []
        for _ in range(num_games):
            result.append(int(self.play_single_game()))
        result = [(i + 1) // 2 for i in result]
        print(f'Win: {(sum(result) / num_games) * 100}%')


if __name__ == '__main__':
    a = Agent()
    a.play_num_games(num_games=50)
