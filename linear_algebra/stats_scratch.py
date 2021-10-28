exit()
python3
print("\n"*100)
import linalg as la

A = la.Mat([[1, 1, 3],
            [2, 2, 4],
            [2, 0, -1]])

B = la.Mat([[1, 1, 1],
            [0, 2, 0],
            [0, 3, -4]])

C = la.stats.gen_centering([3,3])

# la.print_mat(la.stats.gen_centering(5),2)
la.print_mat(C.multiply(B),2)

