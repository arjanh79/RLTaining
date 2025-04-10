
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