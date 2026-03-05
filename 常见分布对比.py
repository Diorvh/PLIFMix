from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
# 比较不同分布的形状
distributions = {
    'Normal': stats.norm(),
    'Laplace': stats.laplace(),
    'Logistic': stats.logistic(),
    't (df=5)': stats.t(5),
    'Cauchy': stats.cauchy()
}

x = np.linspace(-10, 10, 1000)
plt.figure(figsize=(10, 6))
for name, dist in distributions.items():
    plt.plot(x, dist.pdf(x), label=name)

plt.legend()
plt.title('中心对称分布比较')
plt.show()