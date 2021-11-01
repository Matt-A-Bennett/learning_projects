exit()
python3
print("\n"*100)
import linalg as la

A = la.Mat([[1, 2, 3],
            [2, 3, 1],
            [3, 3, 0],
            [4, 5, 1],
            [5, 6, -1]])

res, df = la.stats.ttest(A, axis=0)
la.print_mat(res, 4)
