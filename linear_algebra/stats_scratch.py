exit()
python3
print("\n"*100)
import linalg as la

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

la.stats.ttest2(u, v)




