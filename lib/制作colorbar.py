import cv2
import numpy as np
from lib_cv import get_modified_jet

def create_colorbar(height, width, colormap, scale_min, scale_max, padding_top, padding_bottom):
    # 绘制刻度（考虑10个刻度标记）
    num_steps = 3
    step_size = height / num_steps
    font_scale = 1  # 字体大小
    font_thickness = 2  # 字体粗细

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
        cv2.putText(scale_image, f'{label:.2f}', (width + 10, y + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    return scale_image

# 示例用法
height, width = 768, 40  # 像素高度和宽度
colormap = get_modified_jet()  # 如需更改颜色映射
scale_min, scale_max = 0, 1  # 刻度范围
padding_top, padding_bottom = 20, 20  # 上下留白的长度（像素）

colorbar_with_scale = create_colorbar(height, width, colormap, scale_min, scale_max, padding_top, padding_bottom)

# 显示或保存结果
cv2.imshow('Colorbar with Scale and Padding', colorbar_with_scale)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 可选，保存图像
cv2.imwrite('colorbar_with_scale_and_padding.png', colorbar_with_scale)
