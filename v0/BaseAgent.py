
import torch
import numpy as np

from v0.Environment import Environment, PlotEnvironment


class BaseAgent(Environment):
    def __init__(self, start_x, start_y):
        super().__init__(start_x, start_y)

        self.target_x = torch.tensor(5, dtype=torch.float32)
        self.target_y = torch.tensor(5, dtype=torch.float32)


    def normalize_angle(self, angle):
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi


    def diff_angle_rad(self, current, target):
        return (target - current + np.pi) % (2 * np.pi) - np.pi


    def get_step(self):
        z = self.get_altitude().detach()
        direction, magnitude = self.get_gradiant()
        direction = self.normalize_angle(direction + torch.pi)

        dx_to_target = (self.target_x - self.x).detach().numpy()
        dy_to_target = (self.target_y - self.y).detach().numpy()
        target_angle = np.arctan2(dy_to_target, dx_to_target)

        current_angle = np.arctan2(self.momentum[1].item(), self.momentum[0].item())

        diff_angle = self.diff_angle_rad(current_angle, target_angle)
        diff_angle = self.normalize_angle(diff_angle)
        diff_angle = np.clip(diff_angle, -0.5, 0.5)
        if np.abs(diff_angle) <  0.5:
            magnitude += 1

        direction = self.normalize_angle(direction + diff_angle)
        # print(f'After: {direction} {target_angle}')

        step_vector = (magnitude + self.gravity) * torch.stack([
            torch.cos(direction),
            torch.sin(direction)
        ])

        self.momentum = self.beta * self.momentum + (1 - self.beta) * step_vector

        x = (self.x + self.momentum[0]).detach().requires_grad_()
        y = (self.y + self.momentum[1]).detach().requires_grad_()

        if torch.abs(x) > 15 or torch.abs(y) > 15:
            return

        self.steps.append([x.item(), y.item(), z.item()])
        self.x = x
        self.y = y


ba = BaseAgent(-5, -2.5)
for _ in range(100):
    ba.get_step()
PlotEnvironment(ba.get_results()).plot_mountain('mountain_steer')
PlotEnvironment(ba.get_results()).plot_mountain_2d('mountain_steer')