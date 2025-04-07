import numpy as np
import torch
import matplotlib.pyplot as plt

class Landscape:

    def __init__(self):
        self.peaks = [
            (1.75, 4, 4, 0.05),
            (2.5, -8, 6, 0.04),
            (2.0, -5, -5, 0.04)
        ]

    def generate_landscape(self):
        def landscape_fn(x, y):
            z = torch.zeros_like(x)
            for amplitude, center_x, center_y, sharpness in self.peaks:
                z += amplitude * torch.exp(-sharpness * ((x - center_x) ** 2 + (y - center_y) ** 2))
            z -= 0.00 * torch.sqrt(x ** 2 + y ** 2 + 1e-6)
            return z

        return landscape_fn


class Environment:

    def __init__(self, start_x, start_y):
        self.beta = 0.95
        self.gravity = 1
        self.landscape_fn = Landscape().generate_landscape()

        self.x = torch.tensor(start_x, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(start_y, dtype=torch.float32, requires_grad=True)

        self.momentum = torch.tensor([0.0, 0.0], dtype=torch.float32).detach()

        self.steps = []


    def get_altitude(self):
        return self.landscape_fn(self.x, self.y)


    def get_gradiant(self):
        z = self.get_altitude()
        z.backward()

        dz_dx = self.x.grad
        dz_dy = self.y.grad

        magnitude = torch.sqrt(dz_dx ** 2 + dz_dy ** 2)
        direction = torch.arctan2(dz_dy, dz_dx)

        return direction, magnitude


    def get_step(self):

        z = self.get_altitude()
        direction, magnitude = self.get_gradiant()

        step_vector = (magnitude + self.gravity) * torch.tensor([
            torch.cos(direction),
            torch.sin(direction)
        ])

        self.momentum = self.beta * self.momentum + (1 - self.beta) * step_vector

        x = (self.x - self.momentum[0]).detach().requires_grad_()
        y = (self.y - self.momentum[1]).detach().requires_grad_()

        self.steps.append([x.item(), y.item(), z.item()])
        self.x = x
        self.y = y

    def get_results(self):
        return np.array(self.steps)

    def run(self, num_steps=50):
        for _ in range(num_steps):
            self.get_step()
        PlotEnvironment(self.get_results()).plot_mountain('mountain_free')
        PlotEnvironment(self.get_results()).plot_mountain_2d('mountain_free')


class PlotEnvironment:

    def __init__(self, steps):
        self.landscape_fn = Landscape().generate_landscape()
        self.steps = steps

    def plot_mountain(self, name):
        name += '_3d.png'
        steps = np.array(self.steps)

        x = torch.linspace(-15, 15, 200)
        y = torch.linspace(-15, 15, 200)
        x, y = torch.meshgrid(x, y, indexing='ij')
        z = self.get_altitude_plot(x, y) # Altitude of the mountains

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(steps[:, 0], steps[:, 1], steps[:, 2], color='red', linewidth=1, zorder=10)
        ax.plot_surface(x, y, z, cmap='terrain', edgecolor='none')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.tight_layout()
        plt.savefig(name)

    def get_altitude_plot(self, x, y):
        return self.landscape_fn(x, y)

    def plot_mountain_2d(self, name):
        name += '_2d.png'

        steps = np.array(self.steps)

        x = torch.linspace(-15, 15, 200)
        y = torch.linspace(-15, 15, 200)
        x, y = torch.meshgrid(x, y, indexing='ij')
        z = self.get_altitude_plot(x, y)  # Altitude of the mountains

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.contourf(x, y, z, levels=50, cmap='terrain')
        ax.plot(steps[:, 0], steps[:, 1], color='red', linewidth=1, zorder=10)
        ax.scatter(steps[:, 0], steps[:, 1], color='black', s=20, zorder=11)
        ax.scatter(x=5, y=5, color='red', s=30, zorder=12)


        plt.tight_layout()
        plt.savefig(name)


if __name__ == '__main__':
    Environment(-5, -2.5).run(num_steps=75)