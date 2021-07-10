import torch
import numpy as np
from torch import nn
from torch import optim
from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.solvers import Solver1D
from neurodiffeq.monitors import Monitor1D
import matplotlib.pyplot as plt
import scipy
import math
from mpl_toolkits.mplot3d import Axes3D
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.conditions import IVP   # the initial condition

G = scipy.constants.gravitational_constant
c = scipy.constants.c
M = 1.989 * ((10) ** 30)

# Define the PDE system
# There's only one equation in the system, so the function maps (u, x, y) to a single entry
def R_00(alpha, beta, r): return [
    # np.exp(-2 * (alpha - beta)) *
     (diff(alpha, r, order=2) + diff(alpha, r, order=1) **
                                     2 - diff(alpha, r, order=1)*diff(beta, r, order=1) + 2 * diff(alpha, r, order=1) / r)
]

def R_11(alpha, beta, r): return [
    -1 * (diff(alpha, r, order=2) + diff(alpha, r, order=1)**2 - diff(alpha,
                                                                      r, order=1)*diff(beta, r, order=1) - 2 * diff(beta, r, order=1) / r)
]

def R_22(alpha, beta, r): return [
    # math.exp(2 * beta) *
    (r * (diff(beta, r, order=1) - diff(alpha, r, order=1)) - 1) + 1
]

# Define the boundary conditions
# There's only one function to be solved for, so we only have a single condition
init_vals_pc = [
    IVP(t_0=math.inf, u_0=0.0),
    IVP(t_0=math.inf, u_0=0.0)
]


# Define the neural network to be used
# Again, there's only one function to be solved for, so we only have a single network
nets = [
    FCNN(n_input_units=2, n_output_units=1, hidden_units=[512])
]

monitor = Monitor1D(t_min=(2 * G * M) / (c ** 2),
                    t_max=math.inf, check_every=100)
monitor_callback = monitor.to_callback()

# Instantiate the solver
solver = Solver1D(
    ode_system=R_00,
    conditions=init_vals_pc,
    t_min=(2 * G * M) / (c ** 2),
    t_max=math.inf,
    # nets=nets,
    # train_generator=Generator2D(
    #     (32, 32), (0, 0), (1, 1), method='equally-spaced-noisy'),
    # valid_generator=Generator2D(
    #     (32, 32), (0, 0), (1, 1), method='equally-spaced'),
)

# Fit the neural network
solver.fit(max_epochs=2000, callbacks=[monitor_callback])

solution = solver.get_solution()

ts = np.linspace((2 * G * M) / (c ** 2), math.inf)
alpha_net, beta_net = solution(ts, to_numpy=True)
print(alpha_net, beta_net)

plt.figure()
plt.plot(ts, alpha_net, label='ANN-based solution of alpha')
plt.plot(ts, beta_net, label='ANN-based solution of beta')
plt.ylabel('u')
plt.xlabel('t')
plt.title('Final Solution')
plt.legend()
plt.show()
