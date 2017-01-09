'''
This file is solutions about Linear Programming with Semi_smooth method.
'''

# Auther: Zhang David <pkuzc@pku.edu.cn>

class semi_smooth():
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
		self.invcov = np.linalg.inv(self.cov)
		self.beta = np.eye(self.n) - np.dot(self.A.T, np.dot(self.invcov, self.A))
		self.alpha = np.dot(self.A.T, np.dot(self.invcov, self.b))
		self.step_size = 1.0 / np.linalg.norm(self.cov, 2)
		self.t = t
		self.tau = 0.5
		self.v = 0.9
		self.eta1 = 0.1
		self.eta2 = 0.9
		self.gamma1 = 1.5
		self.gamma2 = 10.0
		self.lambda_ = 1e-2
		self.obj_path = []
		self.cond_path = []

	def prox_tf(self, x):
		return np.dot(self.beta, x) + self.alpha

	def F_Jacob(self, z):
		inner_proj = z - self.t * self.c
		proj = inner_proj[:]
		proj[inner_proj<=0.] = 0.
		f = proj - self.prox_tf(2. * proj - z)
		dia = np.zeros_like(inner_proj)
		dia[inner_proj>=0.] = 1.
		mydiag = np.diag(dia)
		J = mydiag - np.dot(self.beta, 2.0 * mydiag - np.eye(self.n))
		return f, J

	def update(self, z, u_, lambdak):
		f_value, jacob = self.F_Jacob(z)
		muk = lambdak * np.linalg.norm(f_value)
		JpMu = jacob + 1.0 * muk * np.eye(self.n) 
		dk = np.dot(np.linalg.inv(JpMu), -f_value)
		uk = z + dk
		f_uk, _ = self.F_Jacob(uk)
		rho_k = - np.dot(f_uk, dk) / np.linalg.norm(dk) ** 2
		f_u_, _ = self.F_Jacob(u_)
		cond1 = rho_k >= self.eta1
		cond2 = np.linalg.norm(f_uk) <= 1.0 * self.v * np.linalg.norm(f_u_)
		if cond1 and cond2:
			z = uk
			u_ = uk
		elif cond1 and not cond2:
			vk = z - (np.dot(f_uk, z-uk)/np.linalg.norm(f_uk)**2) * f_uk
			z = vk
		else:
			pass

		if rho_k >= self.eta2:
			lambdak = (self.lambda_ + lambdak) / 2.
		elif self.eta1 <= rho_k and rho_k < self.eta2:
			lambdak = (lambdak + self.gamma1 * lambdak) / 2.
		else:
			lambdak = (self.gamma1 * lambdak + self.gamma2 * lambdak) / 2.

		return z, u_, lambdak

	def train(self):
		import time
		start_time = time.time()
		print 'Semi_smooth' + ' is Solving...'
		self.z = 1.0 * np.zeros(self.n)
		self.lambdak = 1.
		self.u_ = self.z[:]
		self.x = self.z - self.t * self.c
		self.x[self.x<0.] = 0.
		self.condition = np.sum((np.dot(self.A, self.x) - self.b) ** 2)
		self.solve = np.dot(self.c, self.x)
		self.err_rate = (self.solve - self.tol2) / np.abs(self.tol2)
		self.iters = 0
		while((self.condition > self.tol1 or self.err_rate >= 1e-3) and self.iters < self.iteration):
			self.z, self.u_, self.lambdak = self.update(self.z, self.u_, self.lambdak)
			self.x = self.z - self.t * self.c
			self.x[self.x<0.] = 0.
			self.condition = np.sum((np.dot(self.A, self.x) - self.b) ** 2)
			self.solve = np.dot(self.c, self.x)
			self.err_rate = np.abs((self.solve - self.tol2) / self.tol2)
			self.iters += 1
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

	sm_smooth = semi_smooth(c, A, b, 0.1, 2000, tol1=cvx_condition, tol2=prob.solve())
	sm_smooth.train()
	print 'model run_time:', sm_smooth.run_time
	print 'model objective:', sm_smooth.solve
	print 'model condition:', sm_smooth.condition


