class SGD_variants():
	import numpy as np
	def __init__(self, A, b, mu, init_iteration, max_iteration, tol):
		self.A = A
		self.AT = self.A.T
		self.b = b
		self.m, self.n = self.A.shape
		self.mu = mu 
		self.init_iteration = init_iteration
		self.max_iteration = max_iteration
		self.tol = tol
		self.ATb = np.dot(self.A.T, self.b)
		self.cov = np.dot(self.AT, self.A)
		self.step_size = 1.0/np.linalg.norm(self.cov, 2)
		self.result_path = []
		self.x = 1.0 * np.zeros(self.n)

	def loss(self, x):
		x = x.reshape(-1)
		return 0.5 * np.sum(np.square(np.dot(A, x) - b)) + mu * np.sum(np.abs(x))

	def prox(self, u, t):
		y = np.maximum(np.abs(u) - t, 0)
		y = np.sign(u) * y
		return y

	def train(self, method="Momentum"):
		import time
		start_time = time.time()
		print method + ' is Solving...'

		if method == "Momentum":
			x_ = 1.0 * np.zeros_like(self.x)
			delta_x = self.x - x_
			alpha = 0.9

			for hot_mu in [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
				for i in xrange(self.init_iteration):
					x_ = 1.0 * self.x[:]
					grad = 1.0 * (np.dot(self.cov, self.x) - self.ATb)
					self.x = self.prox(self.x - self.step_size * grad + alpha * delta_x, self.step_size * hot_mu)
					delta_x = self.x - x_
					self.result_path.append(self.loss(self.x))

			self.iters = 1
			self.err_rate = 1.0
			while(self.err_rate > self.tol and self.iters < self.max_iteration):
				self.result_path.append(self.loss(self.x))
				x_ = 1.0 * self.x[:]
				grad = 1.0 * (np.dot(self.cov, self.x) - self.ATb)
				self.x = self.prox(self.x - self.step_size * grad + alpha * delta_x, self.step_size * self.mu)
				self.result_path.append(self.loss(self.x))
				self.err_rate = np.abs(self.loss(self.x)-self.loss(x_))/self.loss(x_)
				self.iters += 1

		elif method == 'AdaGrad':
			eta = 1.0
			G = np.zeros_like(self.x)
			for hot_mu in [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
				for i in xrange(self.init_iteration):
					grad = 1.0 * (np.dot(self.cov, self.x) - self.ATb)
					G = G + grad ** 2
					self.x = self.prox(self.x - eta / (np.sqrt(G) + 1e-7) * grad, eta / (np.sqrt(G) + 1e-7) * hot_mu)
					self.result_path.append(self.loss(self.x))

			self.iters = 1
			self.err_rate = 1.0
			while(self.err_rate > self.tol and self.iters < self.max_iteration):
				self.result_path.append(self.loss(self.x))
				x_ = self.x
				grad = 1.0 * (np.dot(self.cov, self.x) - self.ATb)
				G = G + grad ** 2
				self.x = self.prox(self.x - eta / (np.sqrt(G) + 1e-7) * grad, eta / (np.sqrt(G) + 1e-7) * self.mu)
				self.result_path.append(self.loss(self.x))
				self.err_rate = np.abs(self.loss(self.x)-self.loss(x_))/self.loss(x_)
				self.iters += 1

		elif method == 'RMSProp':
			eta = self.step_size
			gamma = 0.5
			v = np.zeros_like(self.x)
			for hot_mu in [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
				for i in xrange(self.init_iteration*5):
					grad = 1.0 * (np.dot(self.cov, self.x) - self.ATb)
					v = gamma * v + (1 - gamma) * (grad ** 2)
					self.x = self.prox(self.x - eta / (np.sqrt(v) + 1e-7) * grad, eta / (np.sqrt(v) + 1e-7) * hot_mu)
					self.result_path.append(self.loss(self.x))

			self.iters = 1
			self.err_rate = 1.0
			while(self.err_rate > self.tol and self.iters < self.max_iteration):
				self.result_path.append(self.loss(self.x))
				x_ = self.x
				grad = 1.0 * (np.dot(self.cov, self.x) - self.ATb)
				v = gamma * v + (1 - gamma) * (grad ** 2)
				self.x = self.prox(self.x - eta / (np.sqrt(v) + 1e-7) * grad, eta / (np.sqrt(v) + 1e-7) * self.mu)
				self.result_path.append(self.loss(self.x))
				self.err_rate = np.abs(self.loss(self.x)-self.loss(x_))/self.loss(x_)
				self.iters += 1

		elif method == 'Adam':
			eta = self.step_size
			gamma1 = 0.9
			gamma2 = 0.9
			m = np.zeros_like(self.x)
			v = np.zeros_like(self.x)
			for hot_mu in [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
				for i in xrange(1, self.init_iteration*2):
					grad = 1.0 * (np.dot(self.cov, self.x) - self.ATb)
					m = gamma1 * m + (1 - gamma1) * grad
					v = gamma2 * v + (1 - gamma2) * (grad ** 2)
					m_hat = m / (1 - gamma1 ** i)
					v_hat = v / (1 - gamma2 ** i)
					self.x = self.prox(self.x - eta / (np.sqrt(v_hat) + 1e-7) * m_hat, eta / (np.sqrt(v_hat) + 1e-7) * hot_mu)
					self.result_path.append(self.loss(self.x))

			self.iters = 1
			self.err_rate = 1.0
			while(self.err_rate > self.tol and self.iters < self.max_iteration):
				self.result_path.append(self.loss(self.x))
				x_ = self.x
				grad = 1.0 * (np.dot(self.cov, self.x) - self.ATb)
				m = gamma1 * m + (1 - gamma1) * grad
				v = gamma2 * v + (1 - gamma2) * (grad ** 2)
				m_hat = m / (1 - gamma1 ** i)
				v_hat = v / (1 - gamma2 ** i)
				self.x = self.prox(self.x - eta / (np.sqrt(v_hat) + 1e-7) * m_hat, eta / (np.sqrt(v_hat) + 1e-7) * self.mu)
				self.result_path.append(self.loss(self.x))
				self.err_rate = np.abs(self.loss(self.x)-self.loss(x_))/self.loss(x_)
				self.iters += 1

		else:
			print "Such method is not yet supported!!"

		self.run_time = time.time() - start_time
		print 'End!'
		
	def plot(self, method='Momentum'):
		from bokeh.plotting import figure, output_file, show
		x = range(len(self.result_path))
		y = self.result_path
		output_file("./sgd_"+method+".html")
		p = figure(title="SGD_"+method, x_axis_label='iteration', y_axis_label='loss')
		p.line(x, y, legend=method, line_width=2)
		show(p)


if __name__ == '__main__':
	import numpy as np
	from bokeh.plotting import figure, output_file, show
	# for reproducibility
	np.random.seed(1337)

	n = 1024
	m = 512
	mu = 1e-3
	init_iteration = int(1e3)
	max_iteration = int(1e3)
	tol = 1e-9

	# Generating test matrices
	A = np.random.normal(size=(m, n))
	u = np.random.normal(size=(n)) * np.random.binomial(1, 0.1, (n)) 
	b = np.dot(A, u).reshape(-1)

	result_time = []
	result_mse = []
	output_file("./SGD_variants.html")
	p = figure(title="SGD_variants", x_axis_label='iteration', y_axis_label='loss')

	for method, color in zip(['Momentum', 'AdaGrad', 'RMSProp', 'Adam'],['orange', 'red', 'blue', 'green']):
		model = SGD_variants(A, b, mu, init_iteration, max_iteration, tol)
		model.train(method)
		result_time.append(model.run_time)
		result_mse.append(np.mean(np.square(model.x-u)))
		x = range(len(model.result_path))
		y = model.result_path
		p.line(x, y, legend=method, line_width=2, line_color=color)

	show(p)


