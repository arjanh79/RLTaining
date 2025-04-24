import torch

import plotly.graph_objects as go
import pandas as pd


class PlotTest:
    def __init__(self):
        pass

    def create_3d_plot(self):
        x = torch.linspace(-1, 1, steps=50)
        y = torch.linspace(-1, 1, steps=50)

        x, y = torch.meshgrid(x, y, indexing='xy')
        z = 0.5 * x ** 2 + 0.5 *  y ** 2


        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

        fig.update_layout(title=dict(text='Test Image'), autosize=False,
                          width=1024, height=1024,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.update_traces(hoverinfo='skip')

        fig.show()

    def create_grad_plot(self):
        x = torch.linspace(-1, 1, steps=100)
        y = torch.linspace(-1, 1, steps=100)

        x, y = torch.meshgrid(x, y, indexing='xy')

        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)

        z = ((0.5 * x ** 2) + (0.5 * y ** 2))
        output = z.sum()
        output.backward()

        angle = torch.arctan2(y.grad, x.grad).detach().numpy()

        x = x.detach().numpy()
        y = y.detach().numpy()
        df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'angle': angle.flatten()})





pt = PlotTest()

pt.create_3d_plot()
pt.create_grad_plot()