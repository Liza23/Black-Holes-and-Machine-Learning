import numpy as np
import matplotlib.pyplot as plt

from neurodiffeq import diff      # the differentiation operation
from neurodiffeq.ode import solve  # the ANN-based solver
from neurodiffeq.conditions import IVP   # the initial condition
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.generators import Generator1D
import torch

def ricci_tensor_one_var(alpha, r): return diff(alpha, r, order=2) + 2 * diff(
    alpha, r, order=1) * diff(alpha, r, order=1) + 2 * (1/r) * diff(alpha, r, order=1)

# specify the initial conditon
init_val = IVP(t_0=3, u_0=10e-5, u_0_prime=0.0)       

solution, loss = solve(
    ode=ricci_tensor_one_var, condition=init_val, t_min=1.2, t_max=5,
    max_epochs=1000,
    monitor=Monitor1D(t_min=1.2, t_max=3, check_every=100),
)

# Plotting Solution
ts = np.linspace(0, 200, 1000)
alpha = solution(ts, to_numpy=True)

plt.figure()
plt.plot(ts, alpha, label='ANN-based solution of $alpha$')
plt.ylabel('alpha')
plt.xlabel('t')
plt.title('Plots')
plt.legend()
plt.show()

# Plotting Loss Functions
ts = np.linspace(0, 2000, 1000)
plt.figure()
plt.plot(ts, loss['valid_loss'])
plt.ylabel('loss')
plt.xlabel('t')
plt.title('Training Loss Function')
plt.show()
