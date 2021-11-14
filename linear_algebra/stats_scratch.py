exit()
python3
print("\n"*100)
import linalg as la

A = la.Mat([[85, 84, 81],
            [86, 83, 92],
            [88, 93, 103],
            [75, 99, 85],
            [78, 97, 97],
            [94, 90, 84],
            [98, 88, 52],
            [79, 90, 88],
            [71, 87, 105],
            [80, 86, 86],
                       ])

u = A.ind('',0)
v = A.ind('',1)
z = A.ind('',2)
z = la.cat(v, z)

ts, df = la.stats.ttest_one_sample(A, H=80)
print(df); la.print_mat(ts, 2)

t, df = la.stats.ttest_paired(u, v)
print(df); print(round(t, 2))

t, df = la.stats.ttest_unpaired(u, z, assume_equal_vars=True)
print(df); print(round(t, 2))

t, df = la.stats.ttest_welchs(u, z)
print(round(df, 2)); print(round(t, 2))

la.print_mat(A.tr())



results = la.stats.anova_one_way(A)
t, df = la.stats.ttest_paired(A.ind('',0), A.ind('',1))
la.print_mat(t)

test = la.Mat([[1, -1, 0, 2],
               [2, 1, 0, 3],
               [0, 1, 1, 4]])
la.print_mat(la.stats.sum(test,0), 2)
la.print_mat(la.stats.mean(test,1), 2)
la.print_mat(la.stats.zero_center(test,0), 2)
la.print_mat(la.stats.covar(test,0), 2)
la.print_mat(la.stats.var(test,1), 2)
la.print_mat(la.stats.sd(test,0), 2)
la.print_mat(la.stats.se(test,1), 2)
la.print_mat(la.stats.zscore(test,1), 2)

A = la.Mat([[1, 2, 3],
            [2, 3, 1],
            [3, 3, 0],
            [4, 5, 1],
            [5, 6, -1]])

la.print_mat(la.stats.corr(A), 2)

u = la.Mat([[1],
            [2],
            [3],
            [3]])

v = la.Mat([[-2],
            [-5],
            [-6],
            [-5]])

u.dot(v)

la.print_mat(la.stats.corr(u,v))
la.print_mat(la.stats.sum(u))
u.size().index(max(u.size()))
ones = la.gen_mat([1,u.size(0)],values=[1])
la.print_mat(ones)
la.print_mat(u)
la.print_mat(ones.multiply(u))

