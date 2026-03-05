import cv2
import numpy as np
from typing import Union, List
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
ZERO_COLOR = [128, 128, 128]
# 类型注解重载，明确不同输入对应的返回类型
def cvimread(path):
    if path.endswith('.npy'):
        return np.load(path)
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
def cvimread_list(*paths):
    return [cvimread(p) for p in paths]

def cvimsave(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imencode('.tif', img)[1].tofile(path)

def BGR2GRAY_float(img):
    return np.dot(img, [0.114, 0.587, 0.299])

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

def BlackedJet_colormap():
    # 生成原始JET颜色表（256个颜色）
    original_jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256,1), cv2.COLORMAP_JET)
    
    # 修改0号位置为黑色 (BGR=0,0,0)
    modified_jet = original_jet.copy()
    modified_jet[0] = ZERO_COLOR  # 第一行对应数值0的颜色
    return modified_jet

def Gray_colormap():
    original_jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256,1), cv2.COLORMAP_JET)
    modified_jet = original_jet.copy()
    for i in range(256):
        modified_jet[i] = [i, i, i]
    return modified_jet


def BlueWhiteRed_colormap():# 蓝色-白色-红色颜色映射表
    """
    创建更平滑的颜色映射表（使用线性插值）
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    # 蓝色到白色部分 (0-128)
    for i in range(0, 129):
        ratio = i / 128.0
        b = 255
        g = int(255 * ratio)
        r = int(255 * ratio)
        colormap[i] = [b, g, r]
    # 白色到红色部分 (129-255)
    for i in range(129, 256):
        ratio = (i - 128) / 127.0
        b = int(255 * (1 - ratio))
        g = int(255 * (1 - ratio))
        r = 255
        colormap[i] = [b, g, r]
    colormap[0] = ZERO_COLOR
    colormap = np.reshape(colormap, (256, 1, 3))
    return colormap

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

def plot_data_advanced(x_data, y_data, width=1308, height=877, dpi=100,plot_style='line', title="数据图表", xlabel="X轴", ylabel="Y轴", axislimit=[(600,750),(0.3,0.7)]):
    """
    参数:
        plot_style: 绘图样式 ('line', 'scatter', 'both')
    """
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # 根据样式绘图
    if plot_style == 'line' or plot_style == 'both':
        ax.plot(x_data, y_data, 'b-', linewidth=2, label='趋势线')
    if plot_style == 'scatter' or plot_style == 'both':
        ax.scatter(x_data, y_data, color='red', s=50, zorder=5, label='数据点')
    
    # 设置图表属性
    ax.tick_params(axis='both', labelsize=20)  # 同时设置x和y轴
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim(axislimit[0])
    ax.set_ylim(axislimit[1])
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    fig.tight_layout()
    
    # 转换为OpenCV格式
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_rgba = np.asarray(buf)
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    return img_bgr

def process_image(img_path, correction_matrix, operation_type):
    # 根据操作类型选择计算方式
    if '.npy' in img_path:
        img = np.load(img_path)
    else:
        img = cvimread(img_path)
    img_float = img.astype(np.float64)
    if operation_type == 'multiply':
        corrected = img_float * correction_matrix  # 乘法[1](@ref)
    elif operation_type == 'subtract':
        corrected = img_float - correction_matrix  # 减法
    # 检查最大值
    max_value = corrected.max()
    print(f"图像 {os.path.basename(img_path)} 修正后最大值：{max_value}")
    # 清理负值
    corrected = np.clip(corrected, 0, None)
    # 保存
    folder = os.path.dirname(img_path)
    name = os.path.splitext(os.path.basename(img_path))[0]
    imsave_float(corrected, os.path.dirname(folder), name)

def plot_scatter_and_line(x1, y1, x2, y2, save_path, title="Scatter and Line Plot", 
                         xlabel="X", ylabel="Y", scatter_label="Origin", 
                         line_label="Fit", figsize=(10, 6), dpi=300):
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制散点图（第一组点）
    plt.scatter(x1, y1, color='blue', alpha=0.7, label=scatter_label, s=5)
    
    # 绘制折线图（第二组点）
    plt.plot(x2, y2, color='red', alpha=0.8, label=line_label, linewidth=2)
    
    # 添加标签和标题
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图片
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()