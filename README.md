# convex_optimization
This is a homework which implements almost all solutions of LASSO, such as cvxpy, gurobi, mosek, gradient descent, proximal primal problem, smoothed primal problem, FISTA, Nesterov second and so on. Currently I completed the Chinese 'README', but in the near future, I will try to write an English version.

为了更好的利用python语言的特性，我将这一部分的代码用类(Class)进行了封装。整体结构的伪代码如下
```python
class grad_method():
    # initial
    def __init__(self, parameters...):
        ...
    # define some preprocess functions,e.g. proximal function and loss function
    def some_function(self, parameters...):
        ...
    # define train 
    def train(self, method="BASIC", parameters...):
        ...
    # plot the loss with iteration
    def plot(self, parameters...):
        ...
```
所以在利用某个方法时，只需要下面第一行代码来构建模型，然后用第二行代码进行训练，（可选择地）用第三行代码进行画图，展示目标函数值(loss)随迭代次数的变化情况。
```python
model = grad_method(parameters...)
model.train(method="BASIC")
model.plot()
```

关于 smoothed 和 proximal 方法，各制作了一幅目标函数随迭代次数变化的图，更清晰地看出算法收敛的快慢。如下：
![Fig1](./proximal_all.png)
可以看到，经过不多的几步后，FISTA 和 Nesterov 的收敛几乎一样了。它们大约经过600步就达到了收敛条件，而一般的 proximal 方法经过大约1600步达到了收敛条件。
放大局部看：
![Fig2](./proximal_part.png)
smoothed 方法的结果与 Proximal 类似，不再展示。
