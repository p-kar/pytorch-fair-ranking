import pdb
import time
import numpy as np
import pulp
from pulp import *
from birkhoff import birkhoff_von_neumann_decomposition
import timeit


def lp_solver(u,constraint):

    N = len(u)
    v = np.log(2) / np.log(np.arange(N) + 2.0)

    if constraint == 'DemoParity':
        f = np.array([1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, -1.0, -1.0, -1.0,-1.0, -1.0, -1.0,-1.0, -1.0, -1.0,-0.1]) / 3.0
    elif constraint == 'DispTreat': 
        f = np.array([1/u[:3].sum(), 1/u[:3].sum(), 1/u[:3].sum(), -1/u[3:].sum(), -1/u[3:].sum(), -1/u[3:].sum()]) / 3
    elif constraint == 'DispImpact':
        f = np.array([u[0]/u[:3].sum(), u[1]/u[:3].sum(), u[2]/u[:3].sum(), -1*u[3]/u[3:].sum(), -1*u[4]/u[3:].sum(), -1*u[5]/u[3:].sum()]) / 3
    g = np.array(v)

    P = [list([]) for i in range(N)]

    for i in range(N):
        for j in range(N):
            P[i].append(pulp.LpVariable('P_{}_{}'.format(i, j), lowBound=0, upBound=1, cat='Continuous'))

    my_lp_problem = pulp.LpProblem('Ranking', pulp.LpMaximize)

    # objective function
    objective = 0
    for i in range(N):
        for j in range(N):
            objective += u[i] * P[i][j] * v[j]
    my_lp_problem += objective, "Z"

    # Constraints
    for i in range(N):
        constraint = 0
        for j in range(N):
            constraint += P[i][j]
        my_lp_problem += constraint == 1.0

    for j in range(N):
        constraint = 0
        for i in range(N):
            constraint += P[i][j]
        my_lp_problem += constraint == 1.0

    constraint = 0
    for i in range(N):
        for j in range(N):
            constraint += f[i] * P[i][j] * g[j]
    my_lp_problem += constraint == 0.0

    start = time.time()
    my_lp_problem.solve()
    end = time.time()
    # print('Status:', pulp.LpStatus[my_lp_problem.status])
    # print('Time taken (s):', end - start)

    Pvals = np.array([var.varValue for var in my_lp_problem.variables()]).reshape(N, N)
    dcg = np.dot(np.exp2(u) - 1.0, np.matmul(Pvals, v))
    # print ('DCG:', dcg)

    start = time.time()
    result = birkhoff_von_neumann_decomposition(Pvals)
    result_max = max(result)
    result_per = result_max[1]
    result_coeff = result_max [0]
    # print result_per
    per_count = len(result)

    end = time.time()
    # print (end-start)
    return dcg, result_per, result_coeff, per_count

if __name__ == "__main__":
    constraint  = 'DemoParity'
    u = np.array([0.81, 0.80, 0.79, 0.78, 0.77, 0.76, 0.81, 0.80, 0.79, 0.78, 0.77, 0.76,0.81, 0.80, 0.79, 0.78, 0.77, 0.76,0.77,0.76])
    dcg, result_per, result_coeff, per_count = lp_solver(u,constraint)
