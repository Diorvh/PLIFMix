import numpy as np
a = np.load(r'D:\工作文件同步\20260416MMHNTO模拟液雾化混合-3\图像处理\底噪.npy')
a = a.ravel()
largest_10_sorted = -np.sort(-a)[:100]
print(largest_10_sorted)