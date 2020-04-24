import numpy as np 

def get_feasible_point(A,B,C):
	indices =  np.where(B<0)[0]
	if len(indices) == 0:
		return np.zeros(C.shape)
	else:
		n,m = A.shape[0], A.shape[1]
		min_b = np.min(B)
		neg_i = np.where(B<0)

		A_ = np.zeros((n+1,m+1))
		A_[:n,:m] = A
		A_[:-m-1,m] = 1.0 
		A_[n,m] = -1.

		B_ = np.zeros(n+1)
		B_[:n] = B
		B_[n] = -min_b + 1

		C_ = np.zeros(m+1)
		C_[m] = 1.

		X_ = np.zeros(m+1)
		X_[m] = min_b

		print(A_,B_,C_,X_)
		X_ = get_solution(A_,B_,C_,X_)

def get_direction(A,B,C,X):
	Z = np.dot(A,X)-B
	indices =  np.where(Z==0)[0]
	T = A[indices]
	print(T)
	T_inv = np.linalg.inv(np.transpose(T))
	alphas = np.dot(T_inv,C)
	
	i = np.where(alphas<0)[0]
	if len(i) == 0:
		return None
	else:
		i = i[0]
		# print(T_inv)
		v = -T_inv[i]
		A_ = A[~np.isin(np.arange(len(A)), indices)]
		B_ = B[~np.isin(np.arange(len(B)), indices)]
		d = (B_ - np.dot(A_,X))/(np.dot(A_,v) + 1e-8)
		d = d[d>=0]
		alpha = np.min(d)
		# print(alpha,v)
		return alpha*v

def print_stats(A,B,C,X):
	print('X is ', X)
	print('C.X is ', np.dot(C,X))
	print('A.X - B is ', np.dot(A,X)-B)
	print('===========================')

def get_solution(A,B,C,X):
	print_stats(A,B,C,X)
	while True:
		V = get_direction(A,B,C,X)
		if V is None:
			break
		else:
			X = X + V
		print_stats(A,B,C,X)
	return X

# A = np.asarray([[3., 2., 5.],
# 			   [2., 1., 1.],
# 			   [1., 1., 3.],
# 			   [5., 2., 4.],
# 			   [-1, 0., 0.],
# 			   [0., -1., 0.],
# 			   [0., 0., -1.]])

# B = np.asarray([55., 26., 30., 57., 0., 0., 0.])
# C = np.asarray([20., 10., 15.])

A = np.asarray([[-1., -1., 0., 1.],
				[3., -10., 0., -4.],
				[1., 3., 0., 0.],
				[0., 1., 1., 0.],
				[0., -1., 0., 1.],
				[0., 0., 1., -1.],
				[-1., 0., 0., 0.],
				[0., -1., 0., 0.],
				[0., 0., -1., 0.],
				[0., 0., 0., -1.]])

B = np.asarray([-9., 158., 779., 45., 13., 200., 0., 0., 0., 0.])
C = np.asarray([13., 1., 9., -1.])

X = get_feasible_point(A,B,C)
X = get_solution(A,B,C,X)

print('Solution is : ',X)