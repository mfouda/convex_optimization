'''
This file is solutions about Linear Programming with augmented lagrangian method.
'''

# Auther: Zhang David <pkuzc@pku.edu.cn>

class alm_lp():
	import numpy as np 
	def __init__(self, c, A, b, rho, iteration, tol1, tol2):
		self.c = c
		self.A = A
		self.b = b
		self.rho = rho
		self.m, self.n = self.A.shape
		self.iteration = iteration
		self.tol1 = tol1
		self.tol2 = tol2
		self.cov = np.dot(self.A, self.A.T)
		self.step_size = 1.0 / np.linalg.norm(self.cov, 2)
		self.obj_path = []
		self.cond_path = []
		
	def gradient(self, y, x):
		proj = x + self.rho * (np.dot(self.A.T, y) - self.c)
		proj[proj<=0.] = 0.
		grad_y = - self.b + np.dot(self.A, proj)
		return grad_y

	def update(self, y, x, method='gradient'):
		for i in range(self.iteration):
			grad_y = self.gradient(y, x)
			y = y - 1. / self.rho * self.step_size * grad_y
		
		x = x + self.rho * (np.dot(self.A.T, y) - self.c)
		x[x<=0.] = 0.
		return y, x

	def train(self, method='gradient'):
		import time
		start_time = time.time()
		print method + ' is Solving...'
		self.y = np.zeros(m)
		self.x = np.zeros(n)
		self.condition = np.sum((np.dot(self.A, self.x) - self.b) ** 2)
		self.solve = np.dot(self.c, self.x)
		self.err_rate = (self.solve - self.tol2) / np.abs(self.tol2)
		while(self.condition > self.tol1 or self.err_rate >= 1e-3):
			self.y, self.x = self.update(self.y, self.x)
			self.condition = np.sum((np.dot(self.A, self.x) - self.b) ** 2)
			self.solve = np.dot(self.c, self.x)
			self.err_rate = np.abs((self.solve - self.tol2) / self.tol2)
			self.obj_path.append(self.solve)
			self.cond_path.append(self.condition)

		self.run_time = time.time() - start_time
		print 'End!'


if __name__ == '__main__':

	import numpy as np
	import time 
	# for reproducibility
	np.random.seed(1337) 

	n = 100
	m = 20
	A = np.random.randn(m, n)
	xs = np.abs(np.random.randn(n) * np.random.binomial(1, 1.0 * m / n, (n)))
	b = np.dot(A, xs)
	y = np.random.randn(m)
	s = np.multiply(np.random.rand(n), (xs==0))
	c = np.dot(A.T, y) + s

	import cvxpy as cvx
	cvx_starttime = time.time()
	x_var = cvx.Variable(n)
	objective = cvx.Minimize(c.T * x_var)
	constraints = [0 <= x_var, A*x_var == b]
	prob = cvx.Problem(objective, constraints)
	cvx_runtime = time.time() - cvx_starttime
	print 'CVX objective:', prob.solve()
	x_cvx = np.asarray(x_var.value).reshape(-1)
	cvx_condition = np.sum((np.dot(A, x_cvx) - b) **2)

	alm = alm_lp(c, A, b, 5e-3, 10, tol1=cvx_condition, tol2=prob.solve())
	alm.train()
	print 'model run_time:', alm.run_time
	print 'model objective:', alm.solve
	print 'model condition:', alm.condition




