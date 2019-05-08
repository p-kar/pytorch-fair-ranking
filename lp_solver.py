import pdb
import time
import numpy as np
import pulp
from pulp import *
# from birkhoff import birkhoff_von_neumann_decomposition
import timeit
#import torch

def lp_solver_func(u,Gr,constraint):

    N = u.shape[0]
    #print ("u", u)
    #print ("Gr", Gr)
    v = np.log(2) / np.log(np.arange(N) + 2.0)

    if constraint == 'DemoParity':
        #f = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]) / 10
        f = Gr#/(N/2)
        f[f == 0] = -1
        f = f/(N/2)
        
    else:
        f = Gr.astype(float)
        f[f == 0] = -1
        pos_indices = np.where(Gr == 1)
        neg_indices = np.where(Gr == 0)
        #Group1 Group2 sum
        G1_sum = u[pos_indices].sum()
        G2_sum = u[neg_indices].sum()
        if constraint == 'DispTreat': 
              #f = np.array([1/u[:3].sum(), 1/u[:3].sum(), 1/u[:3].sum(), -1/u[3:].sum(), -1/u[3:].sum(), -1/u[3:].sum()]) / 3     
              f[pos_indices] = f[pos_indices]/G1_sum
              f[neg_indices] = f[neg_indices]/G2_sum
              f = f/(N/2)
        elif constraint == 'DispImpact':
              #f = np.array([u[0]/u[:3].sum(), u[1]/u[:3].sum(), u[2]/u[:3].sum(), -1*u[3]/u[3:].sum(), -1*u[4]/u[3:].sum(), -1*u[5]/u[3:].sum()]) / 3
              #pdb.set_trace()
              f[pos_indices] = f[pos_indices]*u[pos_indices]/G1_sum
              f[neg_indices] = f[neg_indices]*u[neg_indices]/G1_sum
              f = f/(N/2)##################
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
    '''
    for j in range(N):
      constraint = 0
      for i in range(N):
        constraint -= P[i][j]*np.log(P[i][j])
      my_lp_problem += constraint <= 0.1
    #for i in range(N):
    #    constraint = np.max(P[:][i])
    #    my_lp_problem += constraint >= 0.5
    '''

 
    start = time.time()
    my_lp_problem.solve()
    end = time.time()
    # print('Status:', pulp.LpStatus[my_lp_problem.status])
    # print('Time taken (s):', end - start)

    Pvals = np.array([var.varValue for var in my_lp_problem.variables()]).reshape(N, N)
    dcg = np.dot(np.exp2(u) - 1.0, np.matmul(Pvals, v))
    #print ("Pvals:", Pvals)
    #print ("Pvals shape:",np.sum(Pvals,axis = 0))
    #print ("Pvals sums;", np.sum(Pvals,axis = 1))
    # idx_ = np.argmax(Pvals, axis = 0)
    # for x in range(N):
    #     #idx_ = np.argmax(Pvals, axis = 0)
    #     Pvals[x,:] = 0
    #     Pvals[x,idx_[x]] = 1
        
    #print ("Pvals:", Pvals)
    #print ("Pvals shape:",np.sum(Pvals,axis = 0))
    #print ("Pvals sums;", np.sum(Pvals,axis = 1))
    # assert np.sum(Pvals,axis = 0).all() == np.ones((N,1)).all()
    # assert np.sum(Pvals,axis = 1).all() == np.ones((N,1)).all()
    return dcg, Pvals
    '''
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
    '''
if __name__ == "__main__":
    constraint  = 'DemoParity'
    u = np.array([0.81, 0.80, 0.79, 0.78, 0.77, 0.76])
    g = np.array([-1,1,1,1,-1,-1])
    dcg, result_per, result_coeff, per_count = lp_solver_func(u,g,constraint)
    print (dcg, result_per, result_coeff, per_count)
