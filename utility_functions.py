import numpy as np
import cvxpy as cp

def update_policy(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost):
    """Update the policy by solving the convex surrogate problems."""
    x, paras_bar, prob_status_fea = _feasible_update(func_value_np, grad_np, paras_t_np, tau_cost)
    if x == np.inf:
        print('feasible problem break ! status = ', prob_status_fea)

    if x <= 0:
        paras_bar, prob_status_obj = _objective_update(func_value_np, grad_np, paras_t_np,
                                                       tau_reward=tau_reward, tau_cost=tau_cost)
        if paras_bar is None:
            print('objective problem break ! status = ', prob_status_obj)

    return paras_bar


def _objective_update(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost):
    """Solving the surrogate problem corresponding to the objective update"""
    m = grad_np.shape[0] - 1  # number of constraints.
    n = grad_np.shape[1]  # dim of parameter.
    tau_np = tau_cost * np.ones(m + 1)
    tau_np[0] = tau_reward

    paras_cvx = cp.Variable(shape=(n,))
    obj = func_value_np[0] + grad_np[0].T @ (paras_cvx - paras_t_np) + \
          tau_np[0] * cp.sum_squares(paras_cvx - paras_t_np)
    constr = []
    for i in range(1, m + 1):
        constr += [func_value_np[i] + grad_np[i].T @ (paras_cvx - paras_t_np) +
                   tau_np[i] * cp.sum_squares(paras_cvx - paras_t_np) <= 0]
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve(solver=cp.MOSEK)
    paras_mosek = paras_cvx.value

    return paras_mosek, prob.status


def _feasible_update(func_value_np, grad_np, paras_t_np, tau_cost):
    """Solving the surrogate problem corresponding to the feasible update"""
    m = grad_np.shape[0] - 1  # number of constraints.
    n = grad_np.shape[1]  # dim of parameter.
    func_value_np = func_value_np[1:]
    grad_np = grad_np[1:]
    tau_np = tau_cost * np.ones(m)

    paras_cvx = cp.Variable(shape=(n,))
    x_cvx = cp.Variable()
    obj = x_cvx
    constr = []
    for i in range(m):
        constr += [func_value_np[i] + grad_np[i].T @ (paras_cvx - paras_t_np) +
                   tau_np[i] * cp.sum_squares(paras_cvx - paras_t_np) <= x_cvx]
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve(solver=cp.MOSEK)
    x_mosek = prob.value
    paras_mosek = paras_cvx.value

    return x_mosek, paras_mosek, prob.status
