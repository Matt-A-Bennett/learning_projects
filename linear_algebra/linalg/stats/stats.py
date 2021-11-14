# To do:
# generic statistic p values table lookup method (maybe store the tables as
# csv files or dict vars or something)

import linalg.linalg as la
from math import sqrt
from tabulate import tabulate

def sum(A, axis=0):
    # if we have a vector, we sum along it's length
    if min(A.size()) == 1:
        axis = A.size().index(max(A.size()))
    if axis == 1:
        A = A.tr()
    ones = la.gen_mat([1,A.size(0)],values=[1])
    A_sum = ones.multiply(A)
    if axis == 1:
        A_sum = A_sum.tr()
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

def zero_center(A, axis=0):
    if axis == 2:
        global_mean = mean(mean(A)).make_scalar()
        return A.subtract(global_mean)
    elif axis == 1:
        A = A.tr()
    if A.is_square():
        A = gen_centering(A.size()).multiply(A)
    else:
        A_mean_mat = mean(A).tile(axes=[A.size(0),1])
        A = A.subtract(A_mean_mat)
    if axis == 1:
        A = A.tr()
    return A

def covar(A, axis=0, sample=True):
    if axis == 1:
        A = A.tr()
    A_zc = zero_center(A)
    A_cov = A_zc.tr().multiply(A_zc)
    N = A.size(0)
    if sample:
        N -= 1
    A_cov = A_cov.div_elwise(N)
    return A_cov

def var(A, axis=0, sample=True):
    A_covar = covar(A, axis, sample)
    A_var = la.Mat([A_covar.diag()])
    if axis == 1:
        return A_var.tr()
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

def ttest_one_sample(A, axis=0, H=0):
    A_diff = mean(A, axis).subtract(H)
    A_se = se(A, axis, sample=True)
    ts = A_diff.div_elwise(A_se)
    df = A.size(axis)-1
    return ts, df

def ttest_paired(u, v):
    A_diff = u.subtract(v)
    t, df = ttest_one_sample(A_diff)
    return t.make_scalar(), df

def ttest_welchs(u, v):
    diff = mean(u).subtract(mean(v)).make_scalar()
    u_se, v_se = se(u).make_scalar(), se(v).make_scalar()
    t = diff / sqrt(u_se**2 + v_se**2)

    # compute degrees of freedom by Welch–Satterthwaite equation
    Nu, Nv = u.size(0), v.size(0)
    u_sd, v_sd = sd(u).make_scalar(), sd(v).make_scalar()

    df =          ( (u_sd**2/Nu + v_sd**2/Nv)**2 /
    (u_sd**4 / (Nu**2 * (Nu-1)) + v_sd**4 / (Nv**2 * (Nv-1))) )

    return t, df

def ttest_unpaired(u, v, assume_equal_vars=False):
    if not assume_equal_vars:
        return ttest_welchs(u, v)

    diff = mean(u).subtract(mean(v)).make_scalar()
    Nu, Nv = u.size(0), v.size(0)
    u_df, v_df = Nu-1, Nv-1
    df = u_df + v_df
    u_var, v_var = var(u).make_scalar(), var(v).make_scalar()

    pooled_var = (( (u_var * u_df) / (u_df + v_df) ) +
                  ( (v_var * v_df) / (v_df + v_df) ))

    pooled_sd = sqrt(pooled_var)

    t = diff / (pooled_sd * sqrt(1/Nu + 1/Nv))

    return t, df

# Only works for same number of observations per group
def anova_one_way(A):
    n_samp = A.size(0)
    n_groups = A.size(1)
    tot_obs = n_samp*n_groups

    group_means = mean(A)
    group_dev = zero_center(group_means, axis=1)
    sq_dev = group_dev.multiply_elwise(group_dev)

    # sums of squares
    group_SS = sq_dev.multiply_elwise(n_samp)
    SS = sum(group_SS).make_scalar()

    group_SSE = var(A, sample=False).multiply_elwise(n_samp)
    SSE = sum(group_SSE).make_scalar()

    SST = SS + SSE

    # degrees of freedom
    df = A.size(1) - 1
    df_err = tot_obs - A.size(1)
    df_tot = tot_obs - 1

    # mean sum of squares and mean error sum of squares
    MS = SS / df
    MSE = SSE / df_err

    F = MS / MSE

    # strore, print and return results
    results = {'Source':['Treatment', 'Error', 'Total'],
               'Sums of Squares (SS)':[SS, SSE, SST],
               'df':[df, df_err, df_tot],
               'Mean Squares':[MS, MSE],
               'F':[F]}
    print('\n', tabulate(results, headers='keys'))

    return results


