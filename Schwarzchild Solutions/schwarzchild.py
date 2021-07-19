import numpy as np
import matplotlib.pyplot as plt

from neurodiffeq import diff
from neurodiffeq.conditions import IVP 
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.ode import solve_system

# r is replaced here by rho# rho is a dimensionless quantitu

def ricci_tensor(alpha, beta, r): return [
    diff(alpha, r, order=2) + diff(alpha, r, order=1) * diff(alpha, r, order=1) -
    diff(beta, r, order=1) * diff(alpha, r, order=1) +
    2 * (1/r) * diff(alpha, r, order=1),
    diff(alpha, r, order=2) + diff(alpha, r, order=1) * diff(alpha, r, order=1) -
    diff(beta, r, order=1) * diff(alpha, r, order=1) -
    2 * (1/r) * diff(beta, r, order=1)
]

# r -> infi (here 5)
# alpha -> 0
# beta -> 0
init_vals = [
    IVP(t_0=5, u_0=0.0),
    IVP(t_0=5, u_0=0.0)
]

# Obtaining Solutions
solution, loss = solve_system(
    ode_system=ricci_tensor, conditions=init_vals, t_min=1, t_max=5,
    max_epochs=1500,
    monitor=Monitor1D(t_min=1, t_max=5, check_every=100)
)

# Plotting Solution
ts = np.linspace(0, 120, 1000)
alpha, beta = solution(ts, to_numpy=True)

plt.figure()
plt.plot(ts, alpha, label='ANN-based solution of $alpha$')
plt.plot(ts, beta, label='ANN-based solution of $beta$')
plt.ylabel('u')
plt.xlabel('t')
plt.title('Plots for Alpha & Beta')
plt.legend()
plt.show()

# Plotting Loss Functions
ts = np.linspace(0, 1500, 1500)
plt.figure()
plt.plot(ts, loss['train_loss'])
plt.ylabel('loss')
plt.xlabel('t')
plt.title('Training Loss Function')
plt.show()
