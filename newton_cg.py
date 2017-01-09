'''
This file is solutions about Linear Programming with Newton_CG method.
'''

# Auther: Zhang David <pkuzc@pku.edu.cn>

class newton_cg():
	import numpy as np 
	def __init__(self, c, A, b, t, iteration, tol1, tol2):
		self.c = c
		self.A = A
		self.b = b
		self.m, self.n = self.A.shape
		self.iteration = iteration
		self.tol1 = tol1
		self.tol2 = tol2
		self.cov = np.dot(self.A, self.A.T)
		self.step_size = 1.0 / np.linalg.norm(self.cov, 2)
		self.t = t
		self.obj_path = []
		self.cond_path = []

	def L_t(self, y, x):
		l = - np.dot(self.b, y)
		r1_inner = self.t * (np.dot(self.A.T, y) - self.c) + x
		r1_inner[r1_inner<=0.] = 0.
		r = 1.0 / (2 * self.t) * (np.linalg.norm(r1_inner) ** 2 - np.linalg.norm(x) ** 2)
		return l + r

	def NCG(self, y, x, delta):
		mu = 0.25
		eta_bar = 0.5
		tau = 0.5
		tau1 = 0.5
		tau2 = 0.5
		for j in range(3):
			inner_proj = self.t * np.dot(self.A.T, y) - self.t * self.c + x
			dia = 1.0 * np.zeros_like(inner_proj)
			dia[inner_proj>0.] = 1.
			V_j = self.t * np.dot(self.A, np.dot(np.diag(dia), self.A.T))
			projed = inner_proj[:]
			projed[inner_proj<0.] = 0.
			delta_phi = -self.b + np.dot(self.A, projed)
			epsilong_j = tau1 * np.min([tau2, np.linalg.norm(delta_phi)])
			d = - np.dot(np.linalg.inv(V_j + epsilong_j * np.eye(self.m)), delta_phi) 
			i = 0
			cond = self.L_t(y+(delta**i) * d, x) <= self.L_t(y, x)+mu*(delta**i)*np.dot(delta_phi, d) 
			while not cond and i <= self.iteration:
				i += 1
				cond = self.L_t(y+(delta**i) * d, x) <= self.L_t(y, x)+mu*(delta**i)*np.dot(delta_phi, d) 
			y = y + (delta**i) * d

		return y


	def update(self, y, x, delta, delta_0 = 0.8, rho=1.1):
		y = self.NCG(y, x, delta)
		x_inner = x + self.t * (np.dot(self.A.T, y) - self.c)
		x_inner[x_inner<=0.] = 0.
		return y, x_inner, delta

	def train(self):
		import time
		start_time = time.time()
		print 'Newton_CG' + ' is Solving...'
		self.y = np.zeros(self.m)
		self.x = np.zeros(self.n)
		self.delta = 0.5
		self.condition = np.sum((np.dot(self.A, self.x) - self.b) ** 2)
		self.solve = np.dot(self.c, self.x)
		self.err_rate = (self.solve - self.tol2) / np.abs(self.tol2)
		self.iters = 500
		while(self.condition > self.tol1 or self.err_rate >= 1e-3 or self.iters > 0):
			self.y, self.x, self.delta = self.update(self.y, self.x, self.delta)
			self.condition = np.sum((np.dot(self.A, self.x) - self.b) ** 2)
			self.solve = np.dot(self.c, self.x)
			self.err_rate = np.abs((self.solve - self.tol2) / self.tol2)
			self.iters -= 1
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

	ncg = newton_cg(c, A, b, 0.1, 10, tol1=cvx_condition, tol2=prob.solve())
	ncg.train()
	print 'model run_time:', ncg.run_time
	print 'model objective:', ncg.solve
	print 'model condition:', ncg.condition


