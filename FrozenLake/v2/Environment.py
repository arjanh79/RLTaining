
import numpy as np

class Environment:
    def __init__(self, size_x, size_y):
        self.rng = np.random.default_rng()
        self.size_x = size_x
        self.size_y = size_y
        self.lake = self.generate_lake()


    def print_lake(self):
        print(self.lake)

    def get_value(self, x, y):
        return self.lake[x, y]

    def generate_lake(self):
        lake = np.zeros((self.size_x, self.size_y), dtype=int)
        holes = (self.rng.random((self.size_x, self.size_y)) > 0.15) - 1
        lake = lake + holes
        lake[self.size_x - 1, self.size_y - 1] = 1
        return lake


if __name__ == '__main__':
    e = Environment(10, 10)
    e.print_lake()
