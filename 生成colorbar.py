import numpy as np
import cv2 
from lib.lib_cv import *
def create_colorbar(height, width, colormap, scale_min, scale_max, padding_top, padding_bottom):
    # 绘制刻度（考虑10个刻度标记）
    height=height-padding_top-padding_bottom
    num_steps = 4
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
if __name__ == "__main__":
    # 示例用法
    height = 800
    width = 50
    padding_top = 20
    padding_bottom = 20
    scale_min = 0
    scale_max = 4095

    # 创建一个示例颜色映射表（例如，Jet颜色映射）
    colormap = Gray_colormap()# BlueWhiteRed_colormap()

    colorbar_image = create_colorbar(height, width, colormap, scale_min, scale_max, padding_top, padding_bottom)

    cvimsave(colorbar_image, r'D:\工作文件同步\2025.4_环境压力与密度独立变化\英文论文\压力图\4-雾化锥角\colorbar.png')