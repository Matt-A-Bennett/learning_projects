import linalg.linalg as la
from math import sqrt

def sum(A, axis=0):
    # if we have a vector, we sum along it's length
    if min(la.size(A)) == 1:
        axis = la.size(A).index(max(la.size(A)))
    if axis == 1:
        A = A.transpose()
    ones = la.gen_mat([1,la.size(A)[0]],values=[1])
    A_sum = ones.multiply(A)
    return A_sum

def mean(A, axis=0):
    # if we have a vector, we take the mean along it's length
    if min(la.size(A)) == 1:
        axis = la.size(A).index(max(la.size(A)))
    A_sum = sum(A, axis)
    A_mean = A_sum.scale(1/la.size(A)[axis])
    return A_mean

# # NEEDS ADDING TO BLOG
def gen_centering(size):
    if type(size) is int:
        size = [size, size]
    return la.eye(size).subtract(1/size[0])

# NEEDS ADDING TO BLOG (shift each col/row by an amount x, such that the mean=0)
def zero_center(A, axis=0):
    if axis == 2:
        global_mean = mean(mean(A)).data[0][0]
        return A.subtract(global_mean)
    elif axis == 1:
        A = A.transpose()
    if A.is_square():
        A = gen_centering(la.size(A)).multiply(A)
    else:
        A_mean = mean(A)
        ones = la.gen_mat([la.size(A)[0], 1], values=[1])
        A_mean_mat = ones.multiply(A_mean)
        A = A.subtract(A_mean_mat)
    if axis == 1:
        A = A.transpose()
    return A

def covar(A):
    A = zero_center(A)
    A = A.transpose().multiply(A)
    return A

def var(A, pop_or_samp='samp'):
    n_obs = la.size(A)[0]
    A_covar = covar(A)
    A_var = la.Mat([A_covar.diag()])
    if pop_or_samp == 'samp':
        # use inbiased estimator (N-1)
        A_var = A_var.scale(1/(n_obs-1))
    return A_var

def stddev(A, axis=0, pop_or_samp='samp'):
    if axis == 1:
        A = A.transpose()
    if pop_or_samp == 'pop':
        A_var = covar(A)
        A_var = la.Mat([])
    elif pop_or_samp == 'samp':
        A_var = var(A, pop_or_samp=pop_or_samp)
    stddevs = la.function_elwise(A_var, sqrt)
    return stddevs

def stderr(A, axis=0, pop_or_samp='samp'):
    stddevs = stddev(A, axis, pop_or_samp=pop_or_samp)
    stderrs = stddevs.scale(1/sqrt(la.size(A)[axis]))
    return stderrs

# NEEDS IMPLEMENTING
def zscore(A, axis=0):
    pass

def corr(u, v=None):
    if v:
        if la.size(u)[1] != 1:
            u = u.transpose()
        if la.size(v)[1] != 1:
            v = v.transpose()
        u = la.cat(u,v)

    covariance = covar(u)
    stddevs = stddev(u, pop_or_samp='pop')
    stddev_prods = stddevs.transpose().multiply(stddevs)
    corr = div_elwise(covariance, stddev_prods)
    if v:
        corr = corr.data[0][1]
    return corr
