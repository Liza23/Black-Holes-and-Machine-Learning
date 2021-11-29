import numpy as np
import matplotlib.pyplot as plt

from neurodiffeq import diff
from neurodiffeq.conditions import IVP
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.ode import solve
import torch

# ricci tensor equation
# def ricci_tensor(b, r): return diff(b, r, order=2) + diff(b, r, order=1) ** 2 + 2 / r * (1 / r + 2) - 2 * torch.exp(-1 * b) * (1/r ** 2 + 5)
def ricci_tensor(b, r): return r ** 2 * diff(b, r, order=2) + r ** 2 * diff(b, r, order=1) ** 2 + 2 * (1 + 2 * r) - 2 * torch.exp(-1 * b) * (1 + 5 * r ** 2)

# r -> 100
# A(r) -> 8333.34 i.e. B(r) = ln(A(r)) -> 9.028
# A'(r) -> 167.67 i.e. B'(r) = A'(r) / A(r) -> 0.02
init_val = IVP(t_0=100, u_0=9.028, u_0_prime=0.02)

# Obtaining Solutions
solution, loss = solve(
    ode=ricci_tensor, condition=init_val, t_min=1, t_max=100,
    max_epochs=1500,
    monitor=Monitor1D(t_min=1, t_max=10, check_every=100)
)

# Plotting Solution
ts = np.linspace(0, 100, 1000)
alpha = solution(ts, to_numpy=True)
print(alpha)

plt.figure()
plt.plot(ts, alpha, label='ANN-based solution of $B(r) = ln(A(r))$')
plt.ylabel('u')
plt.xlabel('t')
plt.title('Plots for Ricci Tensors')
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