from neurodiffeq.conditions import DirichletBVP2D
from neurodiffeq.solvers import Solver2D
from neurodiffeq.monitors import Monitor2D
from neurodiffeq.generators import Generator2D
import torch

import numpy as np
from torch import nn
from torch import optim
from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.pde import solve2D, Monitor2D
from neurodiffeq.generators import Generator2D, PredefinedGenerator
from neurodiffeq.pde import CustomBoundaryCondition, Point, DirichletControlPoint
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from mpl_toolkits.mplot3d import Axes3D

from neurodiffeq.conditions import IBVP1D
from neurodiffeq.pde import make_animation


R_µν = 1/2 * g_µν * R + 8 * π * k * T_µν

# Define the PDE system


def S(x, g, R, R_uv, R_uv_raised, gamma): return [
    diff(x, order=4)*(R - gamma * R_uv * R_uv_raised + 2 * gamma * R ^ 2)
]


# Define the boundary conditions
# There's only one function to be solved for, so we only have a single condition
conditions = [
    # add conditions for alpha and beta
]

# Define the neural network to be used
# Again, there's only one function to be solved for, so we only have a single network
nets = [
    FCNN(n_input_units=2, n_output_units=1, hidden_units=[512])
]

# Define the monitor callback
monitor = Monitor2D(check_every=10, xy_min=(0, 0), xy_max=(1, 1))
monitor_callback = monitor.to_callback()

# Instantiate the solver
solver = Solver2D(
    pde_system=S,
    conditions=conditions,
    # We can omit xy_min when both train_generator and valid_generator are specified
    xy_min=(0, 0),
    # We can omit xy_max when both train_generator and valid_generator are specified
    xy_max=(1, 1),
    nets=nets,
    train_generator=Generator2D(
        (32, 32), (0, 0), (1, 1), method='equally-spaced-noisy'),
    valid_generator=Generator2D(
        (32, 32), (0, 0), (1, 1), method='equally-spaced'),
)

# Fit the neural network
solver.fit(max_epochs=200, callbacks=[monitor_callback])

# Obtain the solution
solution_neural_net_laplace = solver.get_solution()


def plt_surf(xx, yy, zz, z_label='u', x_label='x', y_label='y', title=''):
    fig = plt.figure(figsize=(16, 8))
    ax = Axes3D(fig)
    surf = ax.plot_surface(xx, yy, zz, rstride=2,
                           cstride=1, alpha=0.8, cmap='hot')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    fig.suptitle(title)
    ax.set_proj_type('ortho')
    plt.show()


xs, ys = np.linspace(0, 1, 101), np.linspace(0, 1, 101)
xx, yy = np.meshgrid(xs, ys)
sol_net = solution_neural_net_laplace(xx, yy, to_numpy=True)
plt_surf(xx, yy, sol_net, title='$u(x, y)$ as solved by neural network')


def solution_analytical_laplace(x, y): return np.sin(
    np.pi*y) * np.sinh(np.pi*(1-x))/np.sinh(np.pi)


sol_ana = solution_analytical_laplace(xx, yy)
plt_surf(xx, yy, sol_net-sol_ana, z_label='residual',
         title='residual of the neural network solution')
