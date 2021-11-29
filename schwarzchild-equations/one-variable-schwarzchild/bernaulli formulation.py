import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook

from neurodiffeq import diff      # the differentiation operation
from neurodiffeq.ode import solve  # the ANN-based solver
from neurodiffeq.conditions import IVP   # the initial condition
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.generators import Generator1D
import torch

def v(alpha, r): return diff(alpha, r, order=1)

def ber(u, r): return diff(u, r, order=1) - 2 + 2 * u / r

# specify the initial conditon
init_val = IVP(t_0=5, u_0=5)       

solution, loss = solve(
    ode=ber, 
    condition=init_val, 
    t_min=1.2, t_max=5,
    max_epochs=1000,
    monitor=Monitor1D(t_min=1.2, t_max=5, check_every=100),
)

# Plotting Solution
ts = np.linspace(0, 200, 1000)
u = solution(ts, to_numpy=True)
v = 1 / u
print(v)

plt.figure()
plt.plot(ts, v, label='ANN-based solution of $v$')
plt.ylabel('u')
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
