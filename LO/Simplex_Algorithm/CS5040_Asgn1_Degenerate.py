# Team Members:
# PUNEET MANGLA - CS17BTECH11029
# SOURADEEP CHATTERJEE - ES17BTECH11028
# ADIL TANVEER - ES17BTECH11026


import numpy as np 
import sys

EPS = 1e-6
MAX_ITER = 10000

def make_non_degenerate(A,B,C):
	iterations = 0;
	while True:
		if (iterations < 1000):
			iterations = iterations + 1
			i = A.shape[0]-A.shape[1]
			B_ = B
			B_[:i] = B[:i] + np.random.uniform(1e-6,1e-5,size=i)
			X = get_feasible_point(A,B_,C)
			Z = np.dot(A,X)-B_
			indices =  np.where(np.abs(Z)<EPS)[0]
			if len(indices) == A.shape[1]:
				print('Degeneracy removed')
				break
		else:
			i = A.shape[0]-A.shape[1]
			B_ = B
			B_[:i] = B[:i] + np.random.uniform(0.1,10,size=i)
			X = get_feasible_point(A,B_,C)
			Z = np.dot(A,X)-B_
			indices =  np.where(np.abs(Z)<EPS)[0]
			if len(indices) == A.shape[1]:
				print('Degeneracy removed')
				break
	return A,B_,C

def get_feasible_point(A,B,C):
	indices =  np.where(B<0)[0]
	if len(indices) == 0:
		return np.zeros(C.shape)
	else:
		for _ in range(50):
			si = np.random.choice(len(A),A.shape[1])
			A_ = A[si]
			B_ = B[si]
			try:
				A_inv = np.linalg.inv(A_)
				X_ = np.dot(A_inv,B_)
				X_indices = np.where(X_==0)[0]
				#print(X_indices)
				Z = np.dot(A,X_) - B
				i = np.where(Z>0)[0]
				if ((len(X_indices)==A_.shape[1]) and (len(i) <= 0)):
					print('Given LP might be infeasible')
					exit()
				elif (len(i)>0):
					continue	
				else :
					return X_
			except np.linalg.LinAlgError:
				continue
		print('Given LP Might be infeasible')
		exit()

def get_direction(A,B,C,X):
	Z = np.dot(A,X)-B
	indices =  np.where(np.abs(Z)<EPS)[0]
	T = A[indices]
	# print(T)
	T_inv = np.linalg.inv(np.transpose(T))
	alphas = np.dot(T_inv,C)
	
	i = np.where(alphas<0)[0]
	if len(i) == 0:
		return None
	else:
		i = i[0]
		# print(T_inv)
		v = -T_inv[i]
		if len(np.where(np.dot(A,v)>0)[0]) == 0:
			print('Given LP is Unbounded')
			exit()
		A_ = A[~np.isin(np.arange(len(A)), indices)]
		B_ = B[~np.isin(np.arange(len(B)), indices)]
		d = (B_ - np.dot(A_,X))/(np.dot(A_,v) + 1e-10)
		d = d[d>=0]
		alpha = np.min(d)
		# print(alpha,v)
		return alpha*v

def print_stats(A,B,C,X):
	print('X is ', X)
	print('C.X is ', np.dot(C,X))
	print('A.X - B is ', np.dot(A,X)-B)
	print('===========================')

def get_solution(A,B,C,X,print_=True):
	i = 0
	if print_:
		print_stats(A,B,C,X)
	while True:
		V = get_direction(A,B,C,X)
		if V is None:
			break
		else:
			X = X + V
		if print_:
			print_stats(A,B,C,X)
	return X

def take_input():
	C = np.asarray(list(map(float, input().split('\t'))))
	B = np.asarray(list(map(float, input().split('\t'))))
	A = []
	temp = 0;
	while (temp<len(B)):
		s = input()
		A.append(np.asarray(list(map(float, s.split('\t')))))
		temp = temp+1
	A = np.asarray(A)
	return A,B,C

A,B,C = take_input()
A,B,C = make_non_degenerate(A,B,C)

X = get_feasible_point(A,B,C)
print('Feasible point is : ',X)
print('=======================')
X = get_solution(A,B,C,X)

print('Solution is : ',X)