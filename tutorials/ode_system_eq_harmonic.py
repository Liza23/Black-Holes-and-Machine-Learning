import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook

from neurodiffeq import diff      # the differentiation operation
from neurodiffeq.ode import solve # the ANN-based solver
from neurodiffeq.conditions import IVP   # the initial condition
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.ode import solve_system

# specify the ODE system
parametric_circle = lambda u1, u2, t : [diff(u1, t) - u2,
                                        diff(u2, t) + u1]
# specify the initial conditions
init_vals_pc = [
    IVP(t_0=0.0, u_0=0.0),
    IVP(t_0=0.0, u_0=1.0)
]

# solve the ODE system
solution_pc, _ = solve_system(
    ode_system=parametric_circle, conditions=init_vals_pc, t_min=0.0, t_max=2*np.pi,
    max_epochs=5000,
    monitor=Monitor1D(t_min=0.0, t_max=2*np.pi, check_every=100)
)

ts = np.linspace(0, 2*np.pi, 100)
u1_net, u2_net = solution_pc(ts, to_numpy=True)
u1_ana, u2_ana = np.sin(ts), np.cos(ts)

plt.figure()
plt.plot(ts, u1_net, label='ANN-based solution of $u_1$')
plt.plot(ts, u1_ana, '.', label='Analytical solution of $u_1$')
plt.plot(ts, u2_net, label='ANN-based solution of $u_2$')
plt.plot(ts, u2_ana, '.', label='Analytical solution of $u_2$')
plt.ylabel('u')
plt.xlabel('t')
plt.title('comparing solutions')
plt.legend()
plt.show()
