import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from lib.lib_5 import *
from lib.lib_cv import *

class ImageDisplay(ttk.Frame):
    def __init__(self, master, root_app, name, save_dir):
        super().__init__(master)
        self.root_app = root_app
        self.name = name
        self.save_dir = os.path.dirname(save_dir)
        self.setup_ui()
        self.image = None
        self.display = None

    def setup_ui(self):
        # 第一行：状态信息
        frm1 = ttk.Frame(self)
        frm1.pack(anchor=tk.W)

        ttk.Label(frm1,text=self.name).pack(side=tk.LEFT)
        self.lbl_info = ttk.Label(frm1, text="坐标: (0, 0) 值: 0")
        self.lbl_info.pack(side=tk.LEFT)

        ttk.Button(frm1, text='保存图像', command=self.save_image).pack(side=tk.LEFT)

        # 第三行：图像显示
        frm3 = ttk.Frame(self)
        frm3.pack(anchor=tk.W)
        self.canvas = tk.Canvas(frm3)
        self.canvas.pack(side=tk.LEFT, expand=False, fill='none')
        self.canvas.bind("<Motion>", self.update_mouse_info)
    
    def update(self, image, gain = 1, colormap = None, zoom = 1, crosshair = None, ellipse = None, denoise = 0, save_name_suffix='', special_points = []):
        self.image = image
        self.gain = gain
        self.crosshair = crosshair
        self.ellipse = ellipse
        self.denoise = denoise
        self.save_name_suffix = save_name_suffix
        self.special_points = special_points
        img = self.image[:]
        # 处理增益
        if self.gain == 0:
            self.gain = 65535 / img.max()
        img = img * self.gain
        # 不是8位就转换8位
        if img.dtype != np.uint8:
            img = img / 256
        # 处理越界数值
        img = np.clip(img, 0, 255).astype(np.uint8)
        # 处理伪彩图
        if colormap is None:
            img = img
        else:
            img = cv2.applyColorMap(img, colormap)
        # 处理边界点
        if self.special_points != []:
            y, x = self.special_points
            img[y, x] = [255, 255, 255]
        # 绘制图形
        if crosshair:
            x, y, LINE_WIDTH = crosshair
            cv2.line(img, (x, 0), (x, img.shape[0]), (0,0,255), LINE_WIDTH)
            cv2.line(img, (0, y), (img.shape[1], y), (0,0,255), LINE_WIDTH)
        if ellipse:
            x, y, a, b, LINE_WIDTH = ellipse
            cv2.ellipse(img, (x, y), (a, b),0, 0, 360, (0,0,255), LINE_WIDTH)
        self.display = img
        # 处理缩放
        h, w = img.shape[:2]
        self.zoom = zoom
        if self.zoom != 1:
            w=int(w/self.zoom)
            h=int(h/self.zoom)
            img = cv2.resize(img, (w, h))
        # 显示到画布上
        self.canvas.config(width=w, height=h)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.delete("all")
        self.canvas.create_image(int(self.canvas['width'])//2,int(self.canvas['height'])//2,image=self.tk_img)
        self.focus_force() # 任意选定焦点

    def update_mouse_info(self, event):
        if self.image is None: return
        # 计算实际图像坐标
        x = int(event.x * self.zoom)
        y = int(event.y * self.zoom)
        value = 0
        if (0 <= x < self.image.shape[1]) and (0 <= y < self.image.shape[0]):
            value = self.image[y, x]
        self.lbl_info.config(text=f"坐标: ({x}, {y}) 值: {value:.2f} 最大值：{self.image.max():.2f}")

    def save_image(self):
        name = self.save_name_suffix
        name += f'-{self.name}'
        if self.gain != 1:
            name += f"-增益{self.gain}"
        if self.crosshair:
            name += f"-十字线{self.crosshair}"
        if self.ellipse:
            name += f"-椭圆{self.ellipse}"
        name += f'-降噪{self.denoise}'
        cvimsave(self.display, os.path.join(self.save_dir+f'\\{self.name}', f"{name}.tif"))