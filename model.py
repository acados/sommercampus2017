from casadi import SX, vertcat

def car_model():
    # car model
    p, v = SX.sym('p'), SX.sym('v')
    a = SX.sym('a')
    rhs = vertcat(v, a)

    # ode definition
    x = vertcat(p, v)
    u = a
    return x, u, rhs
