import cv2
import numpy as np
from typing import Union, List

# 类型注解重载，明确不同输入对应的返回类型
def cvimread(path):
    if path.endswith('.npy'):
        return np.load(path)
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
def cvimread_list(*paths):
    return [cvimread(p) for p in paths]

def cvimsave(img, path):
    cv2.imencode('.tif', img)[1].tofile(path)

def imsave_float(img, folder, name):
    import os
    # 保存npy
    output_path = os.path.join(folder, f"{name}.npy")
    np.save(file=output_path,arr=img)
    # 保存tif
    output_path = os.path.join(folder, f"{name}.tif")
    cvimsave(img.astype(np.uint16), output_path)
    # 保存增益后tif
    GAIN = int(65535 / np.max(img))
    if GAIN == 0:
        GAIN = np.round(65535 / np.max(img), 1)
    output_path = os.path.join(folder, f"{name}-GAIN{GAIN}.tif")
    cvimsave((img * GAIN).astype(np.uint16), output_path)

def get_modified_jet():
    # 生成原始JET颜色表（256个颜色）
    original_jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256,1), cv2.COLORMAP_JET)
    
    # 修改0号位置为黑色 (BGR=0,0,0)
    modified_jet = original_jet.copy()
    modified_jet[0] = [0, 0, 0]  # 第一行对应数值0的颜色
    
    return modified_jet

def create_colorbar(height, width, colormap, scale_min, scale_max, padding_top, padding_bottom):
    # 绘制刻度（考虑10个刻度标记）
    height=height-padding_top-padding_bottom
    num_steps = 5
    step_size = height / num_steps
    font_scale = 1  # 字体大小
    font_thickness = 3  # 字体粗细

    # 创建一个纯灰度渐变图像
    gradient = np.linspace(255, 0, height, dtype=np.uint8)
    gradient = np.tile(gradient, (width, 1)).T  # 转置以使其垂直

    # 应用颜色映射
    colorbar = cv2.applyColorMap(gradient, colormap)

    # 在彩色条上方和下方添加白色留白
    total_height = height + padding_top + padding_bottom
    padded_colorbar = np.full((total_height, width, 3), 255, dtype=np.uint8)
    padded_colorbar[padding_top:padding_top + height, :] = colorbar

    # 添加刻度和标签
    scale_image = np.full((total_height, width + 104, 3), 255, dtype=np.uint8)  # +30像素用于刻度
    scale_image[:, :width] = padded_colorbar  # 将彩色条添加到刻度图像中

    for i in range(num_steps + 1):
        y = int(padding_top + height - i * step_size)
        cv2.line(scale_image, (width, y), (width + 5, y), (0, 0, 0), 2)

        # 添加标签
        label = scale_min + (scale_max - scale_min) * (i / num_steps)
        cv2.putText(scale_image, f'{label:.0f}', (width + 10, y + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    return scale_image