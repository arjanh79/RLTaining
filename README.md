FrozenLake/v1

In reinforcement learning, agents may sometimes fall into repetitive behavior without making meaningful progress — repeatedly visiting the same states within a single episode. To address this, we monitor state visits during each episode. When a state is visited more than once, we interpret this as potential stagnation or local looping.

In response, we temporarily increase ε (epsilon), the exploration rate, to encourage the agent to break free from its current trajectory and discover alternative paths.

Although this mechanism is simple, it has proven surprisingly effective in practice. It helps the agent avoid local optima and promotes broader exploration, ultimately improving long-term performance and policy robustness.
