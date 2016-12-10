class ADMM_method():
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
		self.AAT = np.dot(self.A, self.AT)
		self.ATb = np.dot(self.A.T, self.b)
		self.cov = np.dot(self.AT, self.A)
		self.step_size = 1.0/np.linalg.norm(self.cov, 2)
		self.coef = np.linalg.inv(np.eye(m) + 1.0*self.AAT)
		self.result_path = []

	def loss(self, x):
		x = x.reshape(-1)
		return 0.5 * np.sum(np.square(np.dot(A, x) - b)) + mu * np.sum(np.abs(x))

	def train(self, method="dual"):
		import time
		start_time = time.time()
		print method + ' is Solving...'

		if method == "dual":
			# initial weights
			self.y = np.random.normal(size=(self.m))
			self.z = np.dot(self.AT, self.y)
			self.x = np.zeros(self.n)
			def proj_inf_norm(z, uu):
				v = 1.0*z[:]
				v[z>=uu] = 1.0*uu
				v[z<=-uu] = -1.0*uu
				return v

			def update(y, z, w, uu, t):
				z = np.dot(self.AT, y) + w/t
				z = proj_inf_norm(z, uu)
				y = np.dot(self.coef, self.b + t*np.dot(self.A, z - w/t))
				w = w + t*(np.dot(self.AT, y) - z)
				return y, z, w

			self.iters = 1
			self.err_rate = 1.0
			new_max_iteration = self.max_iteration + 6 * self.init_iteration
			while(self.err_rate > self.tol and self.iters < new_max_iteration):
				self.result_path.append(self.loss(self.x))
				x_ = self.x
				self.y, self.z, self.x = update(self.y, self.z, self.x, self.mu, t=1.0)
				self.err_rate = np.abs(self.loss(self.x)-self.loss(x_))/self.loss(x_)
				self.iters += 1

		elif method == "dal":
			# initial weights
			self.y = np.random.normal(size=(self.m))
			self.z = np.dot(self.AT, self.y)
			self.x = np.zeros(self.n)
			def proj_inf_norm(z, uu):
				v = 1.0*z[:]
				v[z>=uu] = 1.0*uu
				v[z<=-uu] = -1.0*uu
				return v

			def update(y, z, w, uu, t):
				for i in range(2):
					z = np.dot(self.AT, y) + w/t
					z = proj_inf_norm(z, uu)
					y = np.dot(self.coef, self.b + t*np.dot(self.A, z - w/t))
				w = w + t*(np.dot(self.AT, y) - z)
				return y, z, w

			self.iters = 1
			self.err_rate = 1.0
			new_max_iteration = self.max_iteration + 6 * self.init_iteration
			while(self.err_rate > self.tol and self.iters < new_max_iteration):
				self.result_path.append(self.loss(self.x))
				x_ = self.x
				self.y, self.z, self.x = update(self.y, self.z, self.x, self.mu, t=1.0)
				self.err_rate = np.abs(self.loss(self.x)-self.loss(x_))/self.loss(x_)
				self.iters += 1

		elif method == "linear":
			# initial weights
			self.x = np.random.normal(size=(self.n))
			self.y = np.dot(self.A, self.x)
			self.z = np.zeros(self.m)

			def soft_thresholding(x, h):
				y = 1.0*x[:]
				y[x>=h] = 1.0*(y[x>=h] - h)
				y[x<=-h] = 1.0*(y[x<=-h] + h)
				y[np.abs(x)<=h] = 0.0
				return y

			def update(x, y, z, u, t):
				grad = t*np.dot(self.cov, x) - t*np.dot(self.AT, self.b + y - z/t)
				x = soft_thresholding(x - self.step_size * grad, self.step_size * u)
				y = (t*np.dot(self.A, x) + z - t*self.b)/(1.0 + t)
				z = z + t*(np.dot(self.A, self.x) - b - y)
				return x, y, z

			for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
				for k in range(self.init_iteration):
					self.x, self.y, self.z = update(self.x, self.y, self.z, hot_mu, t=1.0)
					self.result_path.append(self.loss(self.x))

			self.iters = 1
			self.err_rate = 1.0
			while(self.err_rate > self.tol and self.iters < self.max_iteration):
				self.result_path.append(self.loss(self.x))
				x_ = self.x
				self.x, self.y, self.z = update(self.x, self.y, self.z, hot_mu, t=1.0)
				self.err_rate = np.abs(self.loss(self.x)-self.loss(x_))/self.loss(x_)
				self.iters += 1

		else:
			print "Such method is not yet supported!!"

		self.run_time = time.time() - start_time
		print 'End!'
		
	def plot(self, method='dual'):
		from bokeh.plotting import figure, output_file, show
		x = range(len(self.result_path))
		y = self.result_path
		output_file("./admm_"+method+".html")
		p = figure(title="ADMM Method_"+method, x_axis_label='iteration', y_axis_label='loss')
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
	output_file("./ADMM.html")
	p = figure(title="ADMM Method", x_axis_label='iteration', y_axis_label='loss')

	for method, color in zip(["dual", "dal", "linear"],["orange", "red", "blue"]):
		model = ADMM_method(A, b, mu, init_iteration, max_iteration, tol)
		model.train(method)
		result_time.append(model.run_time)
		result_mse.append(np.mean(np.square(model.x-u)))
		x = range(len(model.result_path))
		y = model.result_path
		p.line(x, y, legend=method, line_width=2, line_color=color)

	show(p)

	