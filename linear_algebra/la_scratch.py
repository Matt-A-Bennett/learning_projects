exit()
python3

print("\n"*100)
import linalg as la

test = la.Mat([[1, 1, 3],
            [2, 2, 4],
            [5, 2, 6]])

test2 = la.Mat([[1, 2, 3],
            [2, 4, 4],
            [3, 4, 5]])

test = la.Mat([[3, 1, 2],
              [0, 2, 2],
              [0, 0, 1]])

# drop dependent columns
la.print_mat(test.drop_dependent_cols())

test.pivot_sign_code()
test.rank()

l = [1,1,3,2]

la.print_mat(la.gen_mat([0,0]))
la.print_mat(la.Mat([]))

AAt = test.transpose().multiply(test)
la.size(test)
la.print_mat(AAt)

# svd
U, sigma, Vt = test.svd()
la.print_mat(test,3)
la.print_mat(U,3)
la.print_mat(sigma,3)
la.print_mat(Vt,3)
la.print_mat(U.multiply(sigma).multiply(Vt),3)

# positive definiteness
test.is_negdef()
test.is_negsemidef()
test.is_possemidef()
test.is_posdef()
test.pivots()

test.pivot_sign_code()

# diagonalise matrix
evec, eigval, evecinv = test.eigdiag()
la.print_mat(evec,4)
la.print_mat(eigval,4)
la.print_mat(evecinv,4)
la.print_mat(evec.multiply(eigval).multiply(evecinv),5)

# get eigenvalues and eigenvectors
evects, evals = test.eig()
la.print_mat(test)
la.print_mat(evals, 2)
la.print_mat(evects,2)
evect = la.Mat([evects.transpose().data[0]]).transpose()
la.print_mat(evect,2)
la.print_mat(test.multiply(evect),2)
la.print_mat(evect.scale(evals.data[0][0]),2)

b = la.Mat([[0],[-2],[5]])
# [4, 0, -2]

x = A.backsub(b)
la.print_mat(x, 2)
la.print_mat(A.multiply(x))

row = [0,0,0,0]
max(row[0:-1])
row == la.gen_mat([1,4]).data[0]

for idx in range(-1, -(3+1), -1):
    print(idx)


b = la.Mat([[-1],[1],[0]])
(max(b.data) == [0] and min(b.data) == [0])

result = st.zero_center(A)
la.print_mat(result)

result = st.zero_center(A, axis=1)
la.print_mat(result, 2)

test = la.Mat([[1, 1, 3],
         [2, 2, 5],
         [3, 3, 4],
         [4, 4, 4]])
la.print_mat(stats.tot_var(test),2)
la.print_mat(stats.var(test),2)
la.print_mat(stats.var(test),2)
la.print_mat(stats.stddev(test, 1, pop_or_samp='pop'),2)
la.print_mat(stats.stddev(test, 1, pop_or_samp='samp'),2)
la.print_mat(stats.stderr(test, 1, pop_or_samp='pop'),2)
la.print_mat(stats.stderr(test, 1, pop_or_samp='samp'),2)
la.print_mat(stats.corr(test),2)

tmp = stats.multiply_elwise(stddevs,stddevs)
tmp = tmp.scale(3)
la.print_mat(tmp)


la.print_mat(stats.sample_stddev(test),2)
la.print_mat(stats.stderr(test),2)

la.print_mat(stats.corr(test),2)
la.print_mat(stats.stddev(test),2)



test = la.Mat([[1, 2, 3, 4]])

test2 = la.Mat([[1, 2, 3, 4]])
print(stats.corr(test, test2))


test2 = la.Mat([[1, 1, 3],
         [2, 0, 5],
         [4, 3, 4],
         [4, 4, 4]])

la.print_mat(stats.multiply_elwise(test, test2))

# test = la.Mat([[1],[2],[3],[4]])
test.dot(test)
test.length()*test.length()

# la.print_mat(stats.var(test),2)
# la.print_mat(stats.zero_center(test,2),3)
# la.print_mat(stats.zero_center(test,1),3)
# la.print_mat(stats.zero_center(test,0),3)

la.print_mat(stats.sum(test, 1))
la.print_mat(stats.mean(test, 1))

print("\n"*100)
test = la.Mat([[1,3,4,3,6,4,7,8,8,9]])
test2 = la.Mat([[1,2,5,4,6,3,7,7,10,8]])
print(stats.corr(test, test2))


la.print_mat(test.inverse(), 2)


test = la.Mat([[1, 1, 0],
               [2, 1, 0],
               [0, 1, 1]])

test = la.gen_mat([3,3],1)
A,Q,R = test.qr()
la.print_mat(R,2)

print(test.eig())

la.print_mat(test,2)

1, 2, 3, 3,
2, 4, 4, 8,
3, 4, 9, 4,
3, 8, 4, 1

la.print_mat(test,2)

test.is_singular()

test = Mat([[-149, -50, -154],
          [537, 180, 546],
          [-27, -9, -25]])

print_mat(test.multiply(test))
B = gen_mat([1,3],1)

B.length()

size(B)[1]
min(size(B))

print_mat(test.subtract(B))
print_mat(B.scale(2)

C = test.subtract(B.scale(2))

# A = QR
A, Q, R = test.qr()
print_mat(Q, 2)
print_mat(R, 2)
print_mat(A, 13)

# Regression
def quick_plot(b, orders=[1]):
    fig = plt.figure()
    Xs = [i/10 for i in range(len(b.data[0])*10)]
    for idx, order in enumerate(orders):
        fit = b.transpose().polyfit(order=order)
        Ys = []
        for x in Xs:
            y = 0
            for exponent in range(order+1):
                y += fit.data[exponent][0]*x**exponent
            Ys.append(y)

        ax = plt.subplot(1,len(orders),idx+1)
        d = ax.plot(Xs[0::10], b.data[0], '.k')
        f = ax.plot(Xs, Ys, '-r')
    return fig

import matplotlib.pyplot as plt
from random import gauss

a, b, c, d = 100, 2.5, 0.5, 0.02
base = []
for idx, x in enumerate(range(-200, 200)):
    x /= 10
    base.append((a + b*x + c*x**2 + d*x**3)+(gauss(0, 30)))
b = Mat([base])
fig = quick_plot(b, orders=[1,2,3])
plt.show()

# A = PLU
A, P, L, U = test.lu()
print_mat(P)
print_mat(A)
print_mat(L)
print_mat(U)
PL = P.multiply(L)
PLU = PL.multiply(U)
print_mat(PL)
print_mat(PLU)

# inverse
print_mat(test)
print_mat(test.inverse())

# short things from elimination
singular = test.is_singular()
rank = test.rank()
pivots = test.pivots()
det = test.determinant()

# EA = U
test = la.Mat([[1, 1, 2, 2],
              [2, 3, 5, 0],
              [0, 4, 4, 0]])
P, E, A, U, singular, _ = test.elimination()
la.print_mat(test)
la.print_mat(P)
la.print_mat(E)
la.print_mat(A)
la.print_mat(U)

# gen mat
la.print_mat(la.gen_mat([7,7], values=[1], type='upper'))
la.print_mat(la.gen_mat([7,7], values=[2, -1], type='full'))
