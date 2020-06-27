import numpy as np
from scipy.optimize import linprog

"""
Wasserstein距离也被形象地称之为“推土机距离”（Earth Mover's Distance，EMD），因为它可以用一个“推土”的例子来通俗地表达它的含义。

假设在位置i=1,2,…,n处我们分布有p1,p2,…,pn那么多的土，简单起见我们设土的总数量为1，即p1+p2+⋯+pn=1，现在要将土推到位置j=1,2,…,n′上，每处的量为q1,q2,…,qn′，而从i推到j的成本为di,j，求成本最低的方案以及对应的最低成本。
"""
def wasserstein_distance(p, q, D):
    """通过线性规划求Wasserstein距离
    p.shape=[m], q.shape=[n], D.shape=[m, n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    """
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun