import numpy as np
from lib.lib_cv import *
from lib.lib_tkinter import select_file
img = cvimread(select_file())
img[:, :530-1] = 0
img[:, 990-1:] = 0
cvimsave(img, r'D:\工作文件同步\20260416MMHNTO模拟液雾化混合-3\图像处理\标定板-去噪声.tif')