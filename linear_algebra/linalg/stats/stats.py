# To do:
# generic statistic p values table lookup method (maybe store the tables as
# csv files or dict vars or something)

import linalg.linalg as la
from math import sqrt

def sum(A, axis=0):
    # if we have a vector, we sum along it's length
    if min(A.size()) == 1:
        axis = A.size().index(max(A.size()))
    if axis == 1:
        A = A.transpose()
    ones = la.gen_mat([1,A.size(0)],values=[1])
    A_sum = ones.multiply(A)
    if axis == 1:
        A_sum = A_sum.transpose()
    return A_sum

def mean(A, axis=0):
    # if we have a vector, we take the mean along it's length
    if min(A.size()) == 1:
        axis = A.size().index(max(A.size()))
    A_sum = sum(A, axis)
    A_mean = A_sum.div_elwise(A.size(axis))
    return A_mean

def gen_centering(size):
    if type(size) is int:
        size = [size, size]
    return la.eye(size).subtract(1/size[0])

# NEEDS ADDING TO BLOG (new tile)
def zero_center(A, axis=0):
    if axis == 2:
        global_mean = mean(mean(A)).make_scalar()
        return A.subtract(global_mean)
    elif axis == 1:
        A = A.transpose()
    if A.is_square():
        A = gen_centering(A.size()).multiply(A)
    else:
        A_mean_mat = mean(A).tile(axes=[A.size(0),1])
        A = A.subtract(A_mean_mat)
    if axis == 1:
        A = A.transpose()
    return A

def covar(A, axis=0, sample=True):
    if axis == 1:
        A = A.transpose()
    A_zc = zero_center(A)
    A_cov = A_zc.transpose().multiply(A_zc)
    N = A.size(0)
    if sample:
        N -= 1
    A_cov = A_cov.div_elwise(N)
    return A_cov

def var(A, axis=0, sample=True):
    A_covar = covar(A, axis, sample)
    A_var = la.Mat([A_covar.diag()])
    if axis == 1:
        return A_var.transpose()
    return A_var

def sd(A, axis=0, sample=True):
    A_var = var(A, axis, sample)
    sds = A_var.function_elwise(sqrt)
    return sds

def se(A, axis=0, sample=True):
    sds = sd(A, axis, sample)
    N = A.size(axis)
    ses = sds.div_elwise(sqrt(N))
    return ses

# NEEDS ADDING TO BLOG (new tile)
def zscore(A, axis=0, sample=False):
    A_zc = zero_center(A, axis)
    A_sd = sd(A_zc, axis, sample)
    if axis == 1:
        A_sd_rep = A_sd.tile(axes=[1, A.size(1)])
    else:
        A_sd_rep = A_sd.tile(axes=[A.size(0), 1])
    return A_zc.div_elwise(A_sd_rep)

def corr(A, axis=0):
    K = covar(A, axis)
    sds=[1/sqrt(x) for x in K.diag()]
    K_sqrt = la.gen_mat([len(sds)]*2, values=sds, type='diag')
    correlations = K_sqrt.multiply(K).multiply(K_sqrt)
    return correlations

# NEEDS ADDING TO BLOG
def ttest(A, axis=0, H=0):
    A_diff = mean(A, axis).subtract(H)
    A_se = se(A, axis, sample=True)
    ts = A_diff.div_elwise(A_se)
    df = A.size(axis)
    return ts, df

# NEEDS ADDING TO BLOG
def ttest2(u, v):
    diff = mean(u).subtract(mean(v)).make_scalar()
    Nu, Nv = u.size(0), v.size(0)
    u_var, v_var = var(u).make_scalar(), var(v).make_scalar()
    t = diff / (                                    # mean difference of groups
        sqrt(                                       # standard error of diff =
            ((u_var * (Nu-1)) + (v_var * (Nv-1))) / # pooled standard deviation
                      (Nu + Nv - 2)) *              #   "       "         "
                    sqrt(1/Nu + 1/Nv))              # 1 / degrees of freedom

    return t

