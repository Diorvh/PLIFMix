import cv2
import glob
import numpy as np
from PIL import Image
import os
from lib.lib_cv import *

# 图片文件夹路径
# NAME=13
GAIN = 2.0
# folder_path = r'I:\20260508-素材制作\100 ms'
# folder_path = r'I:\20260508-素材制作\10 ms\新建文件夹'
# folder_path = r'I:\20260508-素材制作\TIF'
folder_path = r'D:\工作文件同步\杰青申报\ppt\2026杰青\实验编号43'

# 读取所有tif文件
img_paths = os.listdir(folder_path)

# 读取图像并转换为PIL格式
images = []
img_paths = img_paths[100:200]

for file in img_paths:
    # 使用cv2读取图像
    file_path = os.path.join(folder_path, file)
    # img = cv2.imdecode(np.fromfile(file=file_path, dtype=np.uint8), -1)   
    img = cvimread(file_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.clip(img, 0, 255)
    # img = cv2.applyColorMap(img, get_modified_jet())
    # 转换为PIL图像格式
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)  # 将16位图像转换为8位
    img = np.clip(img*GAIN, 0, 255).astype(np.uint8)  # 增强对比度
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    # img = img.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    images.append(img)

# 等比例缩放到宽度不超过640像素（可选）
max_width = 400
for i in range(len(images)):
    if images[i].width > max_width:
        ratio = max_width / images[i].width
        new_size = (max_width, int(images[i].height * ratio))
        images[i] = images[i].resize(new_size, Image.LANCZOS)
# 设置duration为33毫秒（30帧/秒）
duration = 50  # 这里设置为帧率30帧/秒

# 保存为GIF
gif_path = os.path.join(os.path.dirname(folder_path), 'spray.gif')
images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
print(f"GIF saved to {gif_path}")