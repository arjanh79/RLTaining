from FrozenLake.v1.Agent import Agent

import numpy as np

a = Agent()

for _ in range(50):
    a.train(50)
print(np.round(a.Q_table, 2))
a.get_best_route()
