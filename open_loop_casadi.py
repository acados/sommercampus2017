import matplotlib.pyplot as plt
from numpy import array, diag

from casadi import horzcat, Function, integrator, DM

from model import car_model

from acados import sim_solver

x, u, rhs = car_model()

step = 1.
sim = integrator('sim', 'cvodes', 
                 {'x': x, 'p': u, 'ode': rhs},
                 {'tf': step})

print("sim(x0=[0.1, 0.2], p=0.3)['xf'] =", sim(x0=[0.1, 0.2], p=0.3)['xf'])
input("Press 'Enter' to continue...")

x_k = DM([0, 0])
state_trajectory = x_k
for iteration_no in range(60):
    x_k = sim(x0=x_k, p=3)['xf']
    state_trajectory = horzcat(state_trajectory, x_k)

plt.plot(state_trajectory.full().transpose())
plt.legend(['position p [m],', 'velocity v [m/s]'])
plt.xlabel('time [s]')
plt.show()
