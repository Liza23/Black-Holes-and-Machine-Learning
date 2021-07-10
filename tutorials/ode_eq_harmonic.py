import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook

from neurodiffeq import diff      # the differentiation operation
from neurodiffeq.ode import solve # the ANN-based solver
from neurodiffeq.conditions import IVP   # the initial condition
from neurodiffeq.monitors import Monitor1D

harmonic_oscillator = lambda u, t: diff(u, t, order=2) + u
init_val_ho = IVP(t_0=0.0, u_0=0.0, u_0_prime=1.0)

solution_ho, _ = solve(
    ode=harmonic_oscillator, condition=init_val_ho, t_min=0.0, t_max=2*np.pi,
    max_epochs=3000,
    monitor=Monitor1D(t_min=0.0, t_max=2*np.pi, check_every=100)
)

ts = np.linspace(0, 2*np.pi, 50)
u_net = solution_ho(ts, to_numpy=True)
u_ana = np.sin(ts)

plt.figure()
plt.plot(ts, u_net, label='ANN-based solution')
plt.plot(ts, u_ana, '.', label='analytical solution')
plt.ylabel('u')
plt.xlabel('t')
plt.title('comparing solutions')
plt.legend()
plt.show()