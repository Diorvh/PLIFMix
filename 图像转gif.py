import cv2
import glob
import numpy as np
from PIL import Image
import os
from lib.lib_cv import *

# 图片文件夹路径
NAME=13
folder_path = r'E:\实验数据\荧光测试\20250407孔径比影响-2\原始图像\实验编号'+str(NAME)

# 读取所有tif文件
img_paths = os.listdir(folder_path)[3000:3100]

# 读取图像并转换为PIL格式
images = []
for file in img_paths:
    # 使用cv2读取图像
    file_path = os.path.join(folder_path, file)
    img = cv2.imdecode(np.fromfile(file=file_path, dtype=np.uint8), -1)   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)*2
    img = np.clip(img, 0, 255)
    img = cv2.applyColorMap(img, get_modified_jet())
    # 转换为PIL图像格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    images.append(img_pil)

# 设置duration为33毫秒（30帧/秒）
duration = 100  # 这里设置为帧率30帧/秒
# 保存为GIF
gif_path = os.path.join(os.path.dirname(folder_path), '实验编号'+str(NAME)+'.gif')
images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
print(f"GIF saved to {gif_path}")