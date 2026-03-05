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
        command=lambda: process_images(img_paths, correction_matrix, 'multiply')
    )
    btn_subtract = tk.Button(
        selector, text="减法修正", width=15,
        command=lambda: process_images(img_paths, correction_matrix, 'subtract')
    )
    
    # 布局按钮
    btn_multiply.pack(side=tk.LEFT, padx=20, pady=10)
    btn_subtract.pack(side=tk.RIGHT, padx=20, pady=10)

def process_images(img_paths, correction_matrix,operation_type):
    for img_path in img_paths:
        process_image(img_path, correction_matrix, operation_type)

if __name__ == "__main__":
    # 初始化Tkinter
    root = tk.Tk()
    root.withdraw()
    img_paths = filedialog.askopenfilenames(title="选择待修正文件", filetypes=[("图像矩阵", "*.npy")])
    matrix_path = filedialog.askopenfilename(title="选择修正矩阵文件", filetypes=[("图像矩阵", "*.npy")])
    correction_matrix = np.load(matrix_path)
    show_operation_selector()
    root.mainloop()