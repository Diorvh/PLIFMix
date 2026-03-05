import tkinter as tk
from tkinter import ttk, filedialog
import cv2,os,re
import numpy as np
from PIL import Image, ImageTk
from lib.lib_ConfigManager import ConfigManager
from lib.lib_cv import *
from lib.lib_5 import *
from lib.lib_61 import *

class ImageProcessorApp(ttk.Frame):
    def __init__(self, root, name):
        super().__init__(master=root)
        self.root = root
        self.ConfigManager = ConfigManager(self, name)
        self.initialize_variables()
        self.setup_ui()
        self.ConfigManager.load_config()
        self.process()

    def initialize_variables(self):
        self.original = None
        self.display = None
        self.is_pseudo = True
        self.selecting_center = True
        self.crosshair = True
        self.show_ellipse = True
        self.LINE_WIDTH=5

    def setup_ui(self):
        # 第零行：图片路径选择
        self.fo_path = tk.StringVar()
        self.f_path = tk.StringVar()
        self.o_path = tk.StringVar()
        self.ConfigManager.register_var('fo_path', self.fo_path)
        self.ConfigManager.register_var('f_path', self.f_path)
        self.ConfigManager.register_var('o_path', self.o_path)

        frm0 = ttk.Frame(self)
        frm0.pack(fill=tk.X)
        frm = ttk.Frame(frm0)
        frm.pack(fill=tk.X)
        ttk.Button(frm, text='选择fo图像', command=lambda: self.select_file(self.fo_path)).pack(side=tk.LEFT)
        ttk.Label(frm, textvariable=self.fo_path).pack(side=tk.LEFT)
        frm = ttk.Frame(frm0)
        frm.pack(fill=tk.X)
        ttk.Button(frm, text='选择f图像', command=lambda: self.select_file(self.f_path)).pack(side=tk.LEFT)
        ttk.Label(frm, textvariable=self.f_path).pack(side=tk.LEFT)
        frm = ttk.Frame(frm0)
        frm.pack(fill=tk.X)
        ttk.Button(frm, text='选择o图像', command=lambda: self.select_file(self.o_path)).pack(side=tk.LEFT)
        ttk.Label(frm, textvariable=self.o_path).pack(side=tk.LEFT)

        # 第一行：图片缩放与增益控制
        frm1 = ttk.Frame(self)
        frm1.pack(fill=tk.X)
        
        ttk.Label(frm1, text='全局混合比：').pack(side=tk.LEFT)
        self.ent_globalof = ttk.Entry(frm1, width=5)
        self.ent_globalof.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('globalof', self.ent_globalof)

        ttk.Label(frm1, text="缩小倍数：").pack(side=tk.LEFT)
        self.ent_zoom = ttk.Entry(frm1, width=5)
        self.ent_zoom.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('zoom', self.ent_zoom)

        self.btn_pseudo = ttk.Button(frm1, text="显示伪彩图", command=self.toggle_pseudo)
        self.btn_pseudo.pack(side=tk.LEFT)

        ttk.Label(frm1, text="增益：").pack(side=tk.LEFT)
        self.ent_gain = ttk.Entry(frm1, width=5)
        self.ent_gain.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('gain', self.ent_gain)

        self.save_cbar = tk.BooleanVar()
        ttk.Checkbutton(frm1, text="保存ColorBar", variable=self.save_cbar).pack(side=tk.LEFT)
        self.ConfigManager.register_var('save_cbar', self.save_cbar)

        for ent in ['ent_globalof', 'ent_zoom', 'ent_gain']:
            getattr(self, ent).bind('<Return>', self.process)

        # 第二行：中心点和椭圆控制
        frm2 = ttk.Frame(self)
        frm2.pack(fill=tk.X)

        ttk.Label(frm2, text="中心点坐标：").pack(side=tk.LEFT)
        self.ent_cx = ttk.Entry(frm2, width=5)
        self.ent_cx.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('center_x', self.ent_cx)
        self.ent_cy = ttk.Entry(frm2, width=5)
        self.ent_cy.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('center_y', self.ent_cy)

        self.btn_cross = ttk.Button(frm2, text="显示十字线", command=self.toggle_crosshair)
        self.btn_cross.pack(side=tk.LEFT)
        
        ttk.Label(frm2, text="椭圆长短轴：").pack(side=tk.LEFT)
        self.ent_major = ttk.Entry(frm2, width=5)
        self.ent_major.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('major', self.ent_major)
        self.ent_minor = ttk.Entry(frm2, width=5)
        self.ent_minor.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('minor', self.ent_minor)

        ttk.Label(frm2, text="全图椭圆长短轴：").pack(side=tk.LEFT)
        self.ent_major_whole = ttk.Entry(frm2, width=5)
        self.ent_major_whole.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('major_whole', self.ent_major_whole)
        self.ent_minor_whole = ttk.Entry(frm2, width=5)
        self.ent_minor_whole.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('minor_whole', self.ent_minor_whole)

        self.btn_ellipse = ttk.Button(frm2, text="显示椭圆", command=self.toggle_ellipse)
        self.btn_ellipse.pack(side=tk.LEFT)

        for ent in ['ent_cx', 'ent_cy', 'ent_major', 'ent_minor', 'ent_major_whole', 'ent_minor_whole']:
            getattr(self, ent).bind('<Return>', self.process)

        # 第三行：平移量
        frm3 = ttk.Frame(self)
        frm3.pack(fill=tk.X)

        ttk.Label(frm3, text="f图像左平移量：").pack(side=tk.LEFT)
        self.ent_f_shift = ttk.Entry(frm3, width=5)
        self.ent_f_shift.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('f_shift', self.ent_f_shift)
        ttk.Label(frm3, text="o图像左平移量：").pack(side=tk.LEFT)
        self.ent_o_shift = ttk.Entry(frm3, width=5)
        self.ent_o_shift.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('o_shift', self.ent_o_shift)
        ttk.Button(frm3, text='保存图像', command=self.save_image).pack(side=tk.LEFT)

        self.ent_f_shift.bind('<Return>', self.process)
        self.ent_o_shift.bind('<Return>', self.process)

        # 第四行：结果显示
        frm4 = ttk.Frame(self)
        frm4.pack(fill=tk.X)

        self.lbl_res1 = ttk.Label(frm4,text='计算结果')
        self.lbl_res1.pack(anchor='w')
        self.lbl_res2 = ttk.Label(frm4,text='计算结果')
        self.lbl_res2.pack(anchor='w')

        # 第五行：图像显示
        frm5 = ttk.Frame(self)
        frm5.pack(fill=tk.X)
        
        self.list_display=[]
        self.img_display=[]
        for i in ['f', 'o', 'f+o', 'fo', 'f+o-fo', 'o-f+o']:
            self.list_display.append(ImageDisplay(frm5, self, i, self.base_folder.get()))
            self.list_display[-1].pack(side=tk.LEFT)
    
    def select_file(self, path):
        path.set(filedialog.askopenfilename())
        self.process()
    
    

    def process(self, event = None):
        try:
            # 读取图像
            fo_img = cvimread(self.fo_path.get()).astype(np.float64)
            # f_img = cvimread(exp_plus(self.fo_path.get(), 8)).astype(np.float64)
            # o_img = cvimread(exp_plus(self.fo_path.get(), 16)).astype(np.float64)
            f_img = cvimread(self.f_path.get()).astype(np.float64)
            o_img = cvimread(self.o_path.get()).astype(np.float64)
            # 处理偏移
            f_shift = int(self.ent_f_shift.get())
            o_shift = int(self.ent_o_shift.get())
            # 计算和、差值图像
            combined = shift_image(f_img, f_shift) + shift_image(o_img, o_shift)
            diff = fo_img-combined
            mix_ratio = np.where(combined > 0, shift_image(o_img, o_shift) / combined, 0) * 65536 / float(self.ent_gain.get())# 兼容后续update
            # 更新所有图像
            for idx, (img, shift) in enumerate([(f_img, f_shift), (o_img, o_shift), (combined, 0), (fo_img, 0), (diff, 0), (mix_ratio, o_shift)]):
                self.list_display[idx].update(img, shift)
            # 计算掩模并显示有效数据
            if int(self.ent_major.get()) == 0:
                mask = np.zeros(f_img.shape, dtype=np.uint8)
                height, width = f_img.shape
                new_width = width - 150
                x_start = (width - new_width) // 2
                x_end = x_start + new_width
                y_start = 0
                y_end = height - 1  # OpenCV坐标含结束点
                cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)
                ys, xs = np.where(mask > 0)
                coords_me = (xs, ys)
                coords_abs = (xs, ys)
            else:
                coords_me = generate_mask_coords(int(self.ent_cx.get()), int(self.ent_cy.get()), int(self.ent_major.get()), int(self.ent_minor.get()), f_img.shape)
                coords_abs = generate_mask_coords(int(self.ent_cx.get()), int(self.ent_cy.get()), int(self.ent_major_whole.get()), int(self.ent_minor_whole.get()), f_img.shape)
            abs_val, me1, me2, me3 = calc_result(f_img, o_img, fo_img, f_shift, o_shift, coords_abs, coords_me,  float(self.ent_globalof.get()))
            self.lbl_res1.config(text=f'f_shift: {f_shift}; o_shift: {o_shift}; 差绝对值和{abs_val:.2f}')
            self.lbl_res2.config(text=f'O-FO混合效率: {me1:.4f}; F-FO混合效率: {me2:.4f}; O-F混合效率: {me3:.4f}')
        except Exception as e:
            print(e)

    # 伪彩图切换功能
    def toggle_pseudo(self):
        self.is_pseudo = not self.is_pseudo
        self.btn_pseudo.config(text="关闭伪彩图" if self.is_pseudo else "显示伪彩图")
        self.process()

    # 十字线控制
    def toggle_crosshair(self):
        self.crosshair = not self.crosshair
        self.btn_cross.config(text="关闭十字线" if self.crosshair else "显示十字线")
        self.process()

    # 椭圆控制
    def toggle_ellipse(self):
        self.show_ellipse = not self.show_ellipse
        self.btn_ellipse.config(text="关闭椭圆" if self.show_ellipse else "显示椭圆")
        self.process()

    # 生成图像保存后缀
    def save_image(self):
        for i in self.list_display:
            i.save_image()

def on_close():
    app.ConfigManager.save_config()
    root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root, '6.1')
    app.pack()
    root.mainloop()