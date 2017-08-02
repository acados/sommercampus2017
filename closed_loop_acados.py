import matplotlib.pyplot as plt
from numpy import array, diag
from scipy.linalg import block_diag

from casadi import Function
from acados import ocp_nlp, ocp_nlp_solver

from model import car_model

x, u, rhs = car_model()
f = Function('f', [x, u], [rhs])
print("f([0.1, 0.2], 0.3) =", f([0.1, 0.2], 0.3))
input("Press 'Enter' to continue...")

N = 10
nlp = ocp_nlp({'N': N, 'nx': 2, 'nu': 1, 'nb': [3] + (N-1)*[1] + [0]})

nlp.set_model(f, 1.)
nlp.ub[0] = array([-1000, 0, 5])
nlp.lb[0] = array([-1000, 0, -5])
nlp.idxb[0] = [0, 1, 2]
for i in range(1, N):
    nlp.ub[i] = array([5])
    nlp.lb[i] = array([-5])
    nlp.idxb[i] = [2]

Q = diag([1.0, 1.0])
R = 1e-4
nlp.ls_cost_matrix = N*[block_diag(Q, R)] + [Q]

solver = ocp_nlp_solver('gauss-newton-sqp', nlp)

x0 = array([-1000, 0])
STATES = [x0]
for i in range(50):
    output = solver.evaluate(STATES[-1])
    STATES += [output.states[1]]

plt.ion()
plt.plot([x[0] for x in STATES])
plt.plot([x[1] for x in STATES])
plt.legend(['position p [m],', 'velocity v [m/s]'])
plt.show()
