import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from lib.lib_5 import *
from lib.lib_cv import *

class ImageDisplay(ttk.Frame):
    def __init__(self, master, root_app, name):
        super().__init__(master)
        self.root_app = root_app
        self.name = name
        self.setup_ui()
        self.original = None
        self.display = None
        self.shift = 0

    def setup_ui(self):
        # 第一行：状态信息
        frm2 = ttk.Frame(self)
        frm2.pack(anchor=tk.W)

        ttk.Label(frm2,text=self.name).pack(side=tk.LEFT)
        self.lbl_info = ttk.Label(frm2, text="坐标: (0, 0) 值: 0")
        self.lbl_info.pack(side=tk.LEFT)

        # 第三行：图像显示
        frm3 = ttk.Frame(self)
        frm3.pack(anchor=tk.W)
        self.canvas = tk.Canvas(frm3)
        self.canvas.pack(side=tk.LEFT, expand=False, fill='none')
        self.canvas.bind("<Motion>", self.update_mouse_info)
    
    def update(self, original, shift):
        self.original, self.shift = original, shift
        root = self.root_app
        img = np.abs(original[:])
        # 不是8位就转换8位
        if img.dtype != np.uint8:
            img = img / 256
        # 处理增益
        img = np.clip(img*float(root.ent_gain.get()), 0, 255).astype(np.uint8)
        # 处理伪彩图
        if root.is_pseudo:
            img = cv2.applyColorMap(img.astype(np.uint8), get_modified_jet())
        else:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # 处理平移
        img = shift_image(img, shift)
        # 绘制图形
        if root.crosshair:
            cv2.line(img, (int(root.ent_cx.get()), 0), (int(root.ent_cx.get()), img.shape[0]), (0,0,255), root.LINE_WIDTH)
            cv2.line(img, (0, int(root.ent_cy.get())), (img.shape[1], int(root.ent_cy.get())), (0,0,255), root.LINE_WIDTH)
        if root.show_ellipse:
            cv2.ellipse(img, (int(root.ent_cx.get()), int(root.ent_cy.get())), (int(root.ent_major.get()), int(root.ent_minor.get())),0, 0, 360, (0,0,255), root.LINE_WIDTH)
        self.display = img
        # 处理缩放
        h, w = img.shape[:2]
        self.zoom = float(root.ent_zoom.get())
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
        if self.original is None: return
        # 计算实际图像坐标
        x = event.x * self.zoom
        y = event.y * self.zoom
        # 计算原始数据坐标（考虑平移）
        orig_x = int(x) + self.shift
        orig_y = int(y)
        value = 0
        if (0 <= orig_x < self.original.shape[1]) and (0 <= orig_y < self.original.shape[0]):
            value = self.original[orig_y, orig_x]
        
        self.lbl_info.config(text=f"坐标: ({int(x)}, {int(y)}) 值: {value:.2f} 最大值：{self.original.max():.2f}")

    def save_image(self):
        root = self.root_app
        folder = os.path.dirname(root.fo_path.get())
        name = os.path.basename(root.fo_path.get())
        name += f'-{self.name}'            
        if float(root.ent_gain.get()) != 1:
            name += f"-增益{float(root.ent_gain.get())}"
        if root.is_pseudo:
            name += "-伪彩图"
        name += f"-向左平移{int(root.ent_f_shift.get())}_{int(root.ent_o_shift.get())}"
        if root.crosshair:
            name += f"-十字线{(int(root.ent_cx.get()), int(root.ent_cy.get()))}"
        if root.show_ellipse:
            name += f"-椭圆{(int(root.ent_cx.get()), int(root.ent_cy.get()))}-{(int(root.ent_major.get()), int(root.ent_minor.get()))}"
        # 处理colorbar
        assert self.display is not None
        img = self.display[:].astype(np.uint16)
        if root.save_cbar.get():
            cbar_img= create_colorbar(img.shape[0], 60, get_modified_jet(), 0, 255 / float(root.ent_gain.get()), 50, 50)
            # 拼接图像
            img = np.hstack([img, cbar_img])
            name += f"-伪彩图例"
        output_path = os.path.join(folder, f"{name}.tif")
        cvimsave(img, output_path)