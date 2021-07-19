import numpy as np
import matplotlib.pyplot as plt
import math
from neurodiffeq import diff      # the differentiation operation
from neurodiffeq.ode import solve  # the ANN-based solver
from neurodiffeq.conditions import IVP   # the initial condition
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.ode import solve_system
from neurodiffeq.solvers import Solver1D
import torch

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

solution_pc, loss = solve_system(
    ode_system=ricci_tensor, conditions=init_vals, t_min=1, t_max=5,
    max_epochs=2500,
    monitor=Monitor1D(t_min=1, t_max=5, check_every=200)
)

# monitor = Monitor1D(t_min=1,
#                     t_max=5, check_every=200)
# monitor_callback = monitor.to_callback()


# solver = Solver1D(
#     ode_system=ricci_tensor,
#     conditions=init_vals,
#     t_min=1,
#     t_max=5,
#     # nets=nets,
#     # train_generator=Generator2D(
#     #     (32, 32), (0, 0), (1, 1), method='equally-spaced-noisy'),
#     # valid_generator=Generator2D(
#     #     (32, 32), (0, 0), (1, 1), method='equally-spaced'),
# )

# # # # Fit the neural network
# solver.fit(max_epochs=2000, callbacks=[monitor_callback])

# solution_pc = solver.get_solution()

ts = np.linspace(0, 5, 100)

u1_net, u2_net = solution_pc(ts, to_numpy=True)

print(u1_net, u2_net)

if u1_net.all() + u2_net.all() == 0:
    print("alpha = -beta")

u1_ana, u2_ana = np.sin(ts), np.cos(ts)

plt.figure()
plt.plot(ts, u1_net, label='ANN-based solution of $alpha$')
# plt.plot(ts, u1_ana, '.', label='Analytical solution of $u_1$')
plt.plot(ts, u2_net, label='ANN-based solution of $beta$')
# plt.plot(ts, u2_ana, '.', label='Analytical solution of $u_2$')
plt.ylabel('u')
plt.xlabel('r')
plt.title('comparing solutions')
plt.legend()
plt.show()

# Plotting Loss Functions
ts = np.linspace(0, 2500, 2500)
plt.figure()
plt.plot(ts, loss['train_loss'])
plt.ylabel('loss')
plt.xlabel('t')
plt.title('Training Loss Function')
plt.show()
