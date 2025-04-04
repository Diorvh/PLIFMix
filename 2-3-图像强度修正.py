import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
from lib.lib_cv import *

def show_operation_selector():
    # 创建操作选择窗口
    selector = tk.Toplevel()
    selector.title("选择运算方式")
    
    # 创建按钮并绑定不同操作
    btn_multiply = tk.Button(
        selector, text="乘法修正", width=15,
        command=lambda: process_images('multiply')
    )
    btn_subtract = tk.Button(
        selector, text="减法修正", width=15,
        command=lambda: process_images('subtract')
    )
    
    # 布局按钮
    btn_multiply.pack(side=tk.LEFT, padx=20, pady=10)
    btn_subtract.pack(side=tk.RIGHT, padx=20, pady=10)

def process_images(operation_type):
    for img_path in img_paths:
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
    root.quit()

if __name__ == "__main__":
    # 初始化Tkinter
    root = tk.Tk()
    root.withdraw()
    img_paths = filedialog.askopenfilenames(title="选择待修正文件", filetypes=[("图像矩阵", "*.npy")])
    matrix_path = filedialog.askopenfilename(title="选择修正矩阵文件", filetypes=[("图像矩阵", "*.npy")])
    correction_matrix = np.load(matrix_path)
    show_operation_selector()
    root.mainloop()