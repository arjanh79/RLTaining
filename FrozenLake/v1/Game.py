from FrozenLake.v1.Agent import Agent

import numpy as np

a = Agent()

for _ in range(100):
    a.train(100)
print(np.round(a.Q_table, 2))
a.get_best_route()
