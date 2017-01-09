<h1 align = "center">大作业2</h1>
<h1 align = "center">大数据研究中心 张超 1601214749</h1>



### 理论部分

---

###第1问

a. 改写对偶问题为      
$$
\underset{y,z}{minimize}-b^Ty+l_{\geqslant0} \\
s.t. A^Ty+s=c
$$
则增广拉格朗日函数为
$$
\begin{equation}
\begin{split}
L_t(y,s;x) &= -b^Ty + l_{s \geqslant 0} +  x^T(A^Ty+s-c)+\frac{t}{2}{\parallel A^Ty+s-c \parallel }^2 \\
&=-b^Ty+l_{s \geqslant 0}+\frac{1}{2t}\{{\parallel t(A^Ty+s-c)+x \parallel }^2-{\parallel x \parallel }^2 \}
\end{split}
\end{equation}
$$
其中 $x$ 是对偶问题的拉格朗日乘子，对应原问题中最优解。

则目标转化为求解其最小值：
$$
\hat{y},\hat{s} = \underset{y,s}{argmin}\ \ L_t(y,s;x)
$$
为了消去 $s$ ,我们首先关于 $s$ 求解(3)：
$$
\begin{equation}
\begin{split}
\hat{s} =\underset{s}{argmin} \ \ L_t(y,s;x)
\end{split}
\end{equation}
$$
则需要关于 $s$ 的导数为0，不难求解得到
$$
\hat{s}=\Pi_{s \geqslant 0}(-A^Ty+c-\frac{x}{t})
$$
代入公式(2)中，有
$$
\begin{equation}
\begin{split}
L_t(y,\hat{s};x) &=-b^Ty+\frac{1}{2t}\{\parallel t(A^Ty-c)+x+t\Pi_{s \geqslant 0}(-A^Ty+c-\frac{x}{t})\parallel ^2 - \parallel x\parallel ^2\} \\
&=-b^Ty+\frac{1}{2t}\{\parallel t(A^Ty-c)+x-\Pi_{s \geqslant 0}(t(A^Ty-c)+x))\parallel ^2 - \parallel x\parallel ^2\} \\
&=-b^Ty+\frac{1}{2t}\{\parallel \Pi_{s \geqslant 0}(t(A^Ty-c)+x))\parallel ^2 - \parallel x\parallel ^2\}
\end{split}
\end{equation}
$$
而 $L_t(y,\hat{s};x)​$ 作为 $y​$ 的函数，虽然式子(6)中出现了投影，但是因为外面复合了二次项，所以是关于 $y​$ 的可微函数。故求解最小值，可以使用基于梯度的各种方法。梯度如下：
$$
Grad(y)=\frac{\partial L}{\partial y} = -b+A\Pi_{s \geqslant 0}(t(A^Ty-c)+x))
$$
b. 作业程序中使用了 Gradient Descent方法(alm.py文件)，终止条件(condition)是 $c^Tx$ 与CVXPY的计算结果相差不超过0.1%和$\parallel Ax - b \parallel$小于CVXPY的计算结果。具体步骤如下：

​	Step1: initialize y, x。

​	Step2: $while \ \ not \ \ meet \ \ the \ \ condition:$

​			$while \ \ \Delta L_t(y,\hat{s};x) < \epsilon$:

​				$y = y - step\_size*Grad(y)$

​			$x = \Pi_{s \geqslant 0} (x + t * (A^Ty - c))$

c. 根据参考文献[1]中的算法。我们的子目标为求式子(7)的零点。

首先是算法1，求解方程 $Ax=b$ 的共轭梯度法 $CG(\eta, i_{max})$，具体细节论文中有不再赘述。

算法2，利用算法1，求解方程 $\phi(y)$ 的零点。主体做法为Newton_CG，记为 $NCG(y^0, x, \sigma)$。

算法3，利用算法2，求解 $ L_t(y,\hat{s};x)$ 的极小值点。即：

​		$while \ \ not \ \ meet \ \ the \ \ condition:$

​			$y = NCG(y,x,\sigma)$

​			$x = \Pi_{s \geqslant 0} (x + t * (A^Ty - c))$

​			$\sigma = \rho * \sigma$



### 第2问

a. (1)对偶问题的ADMM:
$$
\begin{equation}
\begin{split}
y^{+} &= \underset{y}{argmin}\ \ L_t(y,s;x) \\
&={(tAA^T)}^{-1}(-Ax-tA(s-c)+b)
\end{split}
\end{equation}
$$

$$
\begin{equation}
\begin{split}
s^{+} &= \underset{s}{argmin}\ \ L_t(y^+,s;x) \\
&=\Pi_{s \geqslant 0}(-\frac{x}{t}+c-A^Ty^+)
\end{split}
\end{equation}
$$

$$
x^+=x+t(A^Ty+s^+-c)
$$

(2)原问题的DRS，首先将原问题转化为：
$$
min \ \ f(x) + h(x) \\
    where： f(x) = l_{Ax-b=0}(x)
    \\ h(x) = c^Tx + l_{x\geqslant 0}(x)
$$
为了使用DRS，我们先求解关于 $f$, $h$ 的 proximal函数。
$$
\begin{equation}
\begin{split}
Prox_{tf}(x) &= \underset{u}{argmin}(f(u)+\frac{1}{2t}\parallel u-x \parallel ^2) \\
&=Proj_{Au=b}(x) \\
&=x+A^T(AA^T)^{-1}(b-Ax) \\
&=(I-A^T(AA^T)^{-1}A)x+A^T(AA^T)^{-1}b \\
&:=\beta x + \alpha
\end{split}
\end{equation}
$$

$$
\begin{equation}
\begin{split}
Prox_{th}(x) &= \underset{u}{argmin}(h(u)+\frac{1}{2t}\parallel u-x \parallel ^2) \\
&=Proj_{x\geqslant 0}(x-tc) 
\end{split}
\end{equation}
$$

所以原问题的DRS算法如下：
$$
\begin{equation}
\begin{split}
x^{+} &= Prox_{th}(z) \\
&=Proj_{x\geqslant 0}(z-tc)
\end{split}
\end{equation}
$$

$$
\begin{equation}
\begin{split}
y^{+} &= Prox_{tf}(2x^+-z) \\
&=2x^+-z+A^T(AA^T)^{-1}(b-A(2x^+-z))
\end{split}
\end{equation}
$$

$$
\begin{equation}
\begin{split}
z^{+} &= z+y^+-x^+\\
&=z+2x^+-z+A^T(AA^T)^{-1}(b-A(2x^+-z))-x^+ \\
&=x^++A^T(AA^T)^{-1}(b-A(2x^+-z))
\end{split}
\end{equation}
$$

从(14~16)可以看出，我们只需要更新 x, z 即可。

b. 对一般性的composite problem进行分析。
$$
\underset{x1,x2}{minimize} \ \ f_1(x_1) + f_2(x_2) \\
s.t. A_1x_1 + A_2x_2 = b
$$
则对偶问题为：
$$
\underset{z}{maximize} \ \ -b^Tz-f_1^*(-A_1^Tz) -f_2^*(-A_2^Tz)
$$
应用 DRS，记
$$
g(z) = b^Tz+f_1^*(-A_1^Tz)  \\
f(z) = f_2^*(-A_2^Tz)
$$
则更新步骤为：
$$
u^+ = prox_{tg}(z+w);  \  \ z^+=prox_{tf}(u^+-w); \  \ w^+ = w+z^+-u^+
$$
注：这是DRS的另外一种表示方法，不难证明这与老师课件中所讲的更新步骤是等价的。

而根据 Moreau decomposition $x = prox_{\lambda f}(x) + \lambda prox_{\lambda ^{-1} f*}(x/\lambda)$，及之前课件中相关证明有：

$u^+ = prox_{tg}(z+w)$ 等价于
$$
\hat{x}_1 = \underset{x_1}{argmin}(f_1(x_1)+z^T(A_1x_1-b)+\frac{t}{2}\parallel A_1x_1-b+\frac{w}{t} \parallel^2) \\
u^+ = z + w+ t ( A_1\hat{x}_1-b)
$$
$z^+=prox_{tf}(u^+-w)=prox_{tf}(z+t ( A_1\hat{x}_1-b))$ 等价于
$$
\hat{x}_2 = \underset{x_2}{argmin}(f_2(x_2)+z^TA_2x_2+\frac{t}{2}\parallel A_1\hat{x}_1+A_2x_2-b \parallel^2) \\
z^+ = z + t ( A_1\hat{x}_1+ A_2\hat{x}_2-b)
$$
最后 $w^+ = w+z^+-u^+$ 等价于
$$
w^+ = tA_2\hat{x}_2
$$
而对于原问题(17)，根据课件中ADMM的更新步骤如下：

![屏幕快照 2017-01-08 上午11.42.48](/Users/pkuzc/Desktop/屏幕快照 2017-01-08 上午11.42.48.png)

显然，两种方法中 $x_1, x_2,z$ 是一致的。

c. 按照 a 中对原问题应用DRS。然后我们的目标变为求解
$$
F(z) = prox_{th}(z) - prox_{tf}(aprox_{th}(z)-z)
$$
的近似零点，即为DRS中的不动点，即达到了收敛条件。

为了下一步编程，应用式子(12)(13)，对 $F(z)$ 进一步转化：
$$
F(z) =\Pi_{s \geqslant 0}(z-tc)-\beta(2\Pi_{s \geqslant 0}(z-tc)-z)-\alpha
$$
Jacobian矩阵为：
$$
J(z) = diag(1_{(z-tc)\geqslant0}) - \beta (2diag(1_{(z-tc)\geqslant0})-I)
$$
其中 
$$
1_{x\geqslant0}=\begin {cases} 1, & x\geqslant{0} \\
0 & x<0 
\end {cases}
$$
然后应用参考文献[2]中所述算法即可。

![屏幕快照 2017-01-08 下午1.14.54](/Users/pkuzc/Desktop/屏幕快照 2017-01-08 下午1.14.54.png)



### 编程实现

------

为了更好的利用python语言的特性，我将这一部分的代码用类(Class)进行了封装。整体结构的伪代码如下:

```python
class specific_method():
    # initial
    def __init__(self, parameters...):
        ...
    # define some preprocess functions,e.g. proximal function and loss function
    def some_function(self, parameters...):
        ...
    # define how to update x in each step
    def update(self, parameters...):
        ...
    # define train 
    def train(self, method="dual", parameters...):
        ...
```
1(b)程序中类名为alm_lp，1(c)中类名为newton_cg，2(a)中类名分别为admm_lp，drs，2(c)中类名为semi_smooth。

### 数值结果

------

实验条件：2.9 GHz处理器，8G内存，python 2.7

| Method      | Time  | $c^Tx$  | $\parallel Ax-b \parallel ^2$ |
| ----------- | :---: | :-----: | :---------------------------: |
| ALM         | 0.60  | -7.8652 |       $2.2*10^{(-26)}$        |
| Newton_CG   | 0.41  | -7.8720 |       $1.7*10^{(-29)}$        |
| ADMM        | 0.72  | -7.8725 |       $1.5*10^{(-24)}$        |
| DRS         | 0.61  | -7.8725 |       $3.3*10^{(-25)}$        |
| Semi_Smooth | 1.81  | -7.8724 |        $2.7*10^{(-7)}$        |
| CVXPY       | 0.001 | -7.8740 |       $2.4*10^{(-24)}$        |

将所用的五种方法的 $c^Tx$ 和 $\parallel Ax-b \parallel ^2$ 分别展示如下：

![obj](/Users/pkuzc/Downloads/obj.png)

![cond](/Users/pkuzc/Downloads/cond.png)

### 结果说明

------

1. 从objective value 图可以看到，Newton_CG和Semi_Smooth 在最开始的几步迭代后就达到比较好的值，且下降速度基本一致；而其他三种方法的结果都慢一些，下降速度基本一致。

   说明：因为Newton_CG 和 Semi_Smooth 都是近似牛顿法的，所以是二阶收敛的。而其他三种是基于一阶梯度的，所以慢一些且下降速度基本一致。

2. objective value 图中，ADMM 与 DRS 的基本上是重合的。

   说明：在第 2 问 b 中我们已经证明了两者的等价性。

3. condition norm value 图中，相比其他几种方法，Semi_Smooth 的波动比较大，且最后结果的 $\parallel Ax-b\parallel ^2$ d的值比较大。

   说明：这是因为其他几种方法的本质都是首先要求 $Ax=b$ 的成立；而 Semi_Smooth 不是。

4. 数值结果中 Semi_Smooth 所花的时间比较长。

   说明：从 objective value 中看到，Semi_smooth 在经过比较少的迭代次数后就达到相对好的值。所以我认为这是因为Semi_Smooth中需要更新的数据量较大，更新步骤较多，且实现步骤中用了比较多的判断条件（关于 $z_k$ 和 $\lambda_k$ 的更新各有 3 个分支），而 python 中条件语句执行的时间代价比较大，所以最后花费的时间比较长。