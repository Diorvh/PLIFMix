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
        self.setup_image_displays()
        self.process()

    def initialize_variables(self):
        self.original = None
        self.display = None
        self.is_pseudo = True
        self.selecting_center = True
        self.crosshair = True
        self.show_ellipse = True
        self.LINE_WIDTH=5
        self.show_contour = False
        self.show_Q_center = False
        self.show_Q_center_fitted = False

    def setup_ui(self):
        # 第零行：图片路径选择
        self.base_folder = tk.StringVar()
        self.ConfigManager.register_var('base_folder', self.base_folder)

        frm0 = ttk.Frame(self)
        frm0.pack(fill=tk.X)
        frm = ttk.Frame(frm0)
        frm.pack(fill=tk.X)
        ttk.Button(frm, text='选择图像文件夹', command=lambda: self.select_folder(self.base_folder)).pack(side=tk.LEFT)
        ttk.Label(frm, textvariable=self.base_folder).pack(side=tk.LEFT)

        frm = ttk.Frame(frm0)
        frm.pack(fill=tk.X)
        ttk.Label(frm, text='fo图像编号：').pack(side=tk.LEFT)
        self.ent_fo_num = ttk.Entry(frm, width=5)
        self.ent_fo_num.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('fo_num', self.ent_fo_num)
        ttk.Label(frm, text='f图像编号：').pack(side=tk.LEFT)
        self.ent_f_num = ttk.Entry(frm, width=5)
        self.ent_f_num.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('f_num', self.ent_f_num)
        ttk.Label(frm, text='o图像编号：').pack(side=tk.LEFT)
        self.ent_o_num = ttk.Entry(frm, width=5)
        self.ent_o_num.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('o_num', self.ent_o_num)
        ttk.Label(frm,text='保存名称').pack(side=tk.LEFT)
        self.ent_save_name = ttk.Entry(frm, width=20)
        self.ent_save_name.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('save_name', self.ent_save_name)

        self.ent_fo_num.bind('<Return>', self.process)
        self.ent_f_num.bind('<Return>', self.process)
        self.ent_o_num.bind('<Return>', self.process)
        self.ent_save_name.bind('<Return>', self.process)


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

        self.btn_ellipse = ttk.Button(frm2, text="显示椭圆", command=self.toggle_ellipse)
        self.btn_ellipse.pack(side=tk.LEFT)

        ttk.Label(frm2, text="线宽").pack(side=tk.LEFT)
        self.ent_line_width = ttk.Entry(frm2, width=5)
        self.ent_line_width.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('line_width', self.ent_line_width)

        for ent in ['ent_cx', 'ent_cy', 'ent_major', 'ent_minor', 'ent_line_width']:
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
        ttk.Label(frm3, text="去噪阈值：").pack(side=tk.LEFT)
        self.ent_denoise = ttk.Entry(frm3, width=5)
        self.ent_denoise.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('denoise', self.ent_denoise)

        self.ent_f_shift.bind('<Return>', self.process)
        self.ent_o_shift.bind('<Return>', self.process)
        self.ent_denoise.bind('<Return>', self.process)

        # 第四行：结果显示
        frm4 = ttk.Frame(self)
        frm4.pack(fill=tk.X)

        self.lbl_res1 = ttk.Label(frm4,text='计算结果')
        self.lbl_res1.pack(anchor='w')
        self.lbl_res2 = ttk.Label(frm4,text='计算结果')
        self.lbl_res2.pack(anchor='w')
        self.lbl_res3 = ttk.Label(frm4,text='计算结果')
        self.lbl_res3.pack(anchor='w')

        # 第五行：轮廓与中心线控制
        frm5 = ttk.Frame(self)
        frm5.pack(fill=tk.X)
        ttk.Label(frm5, text="轮廓流量比例").pack(side=tk.LEFT)
        self.ent_thres_contour = ttk.Entry(frm5, width=5)
        self.ent_thres_contour.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('thres_contour', self.ent_thres_contour)

        self.btn_contour = ttk.Button(frm5, text="显示流量轮廓", command=self.toggle_contour)
        self.btn_contour.pack(side=tk.LEFT)

        
        ttk.Label(frm5, text="流量中心线上下范围").pack(side=tk.LEFT)
        self.ent_thres_Q_center_up = ttk.Entry(frm5, width=5)
        self.ent_thres_Q_center_up.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('thres_Q_center_up', self.ent_thres_Q_center_up)
        self.ent_thres_Q_center_down = ttk.Entry(frm5, width=5)
        self.ent_thres_Q_center_down.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('thres_Q_center_down', self.ent_thres_Q_center_down)
        ttk.Label(frm5, text="流量中心x偏移").pack(side=tk.LEFT)
        self.ent_Q_center_shift = ttk.Entry(frm5, width=5)
        self.ent_Q_center_shift.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('ent_Q_center_shift', self.ent_Q_center_shift)
        ttk.Label(frm5, text="最大值个数").pack(side=tk.LEFT)
        self.ent_max_k = ttk.Entry(frm5, width=5)
        self.ent_max_k.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('max_k', self.ent_max_k)

        self.btn_Q_center = ttk.Button(frm5, text="显示流量中心线", command=self.toggle_Q_center)
        self.btn_Q_center.pack(side=tk.LEFT)
        self.btn_Q_center_fitted = ttk.Button(frm5, text="显示拟合流量中心线", command=self.toggle_Q_center_fitted)
        self.btn_Q_center_fitted.pack(side=tk.LEFT)

        for ent in ['ent_thres_contour', 'ent_thres_Q_center_up', 'ent_thres_Q_center_down', 'ent_Q_center_shift', 'ent_max_k']:
            getattr(self, ent).bind('<Return>', self.process)

    def setup_image_displays(self):
        # 第五行：图像显示
        frm5 = ttk.Frame(self)
        frm5.pack(fill=tk.X)
        
        self.list_display=[]
        self.img_display=[]
        display_title = ['f', 'o', 'fo', 'f+o-fo', 'fo-f-o', 'oDf+o', 'fDfo','oDfo', 'ME_oDf+o', 'ME_fDfo','ME_oDfo']
        display_position = [(0,0),(0,1),(0,2),(0,3),(1,3),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        for j, i in enumerate(display_title):
            self.list_display.append(ImageDisplay(frm5, self, i, self.base_folder.get()))
            self.list_display[-1].grid(row=display_position[j][0], column=display_position[j][1])
    
    def select_folder(self, path):
        path.set(filedialog.askdirectory())
        self.process()

    def process(self, event = None):
        try:
            # 读取图像
            fo_img = cvimread(self.base_folder.get()+f'\\实验编号{self.ent_fo_num.get()}.npy').astype(np.float64)
            f_img = cvimread(self.base_folder.get()+f'\\实验编号{self.ent_f_num.get()}.npy').astype(np.float64)
            o_img = cvimread(self.base_folder.get()+f'\\实验编号{self.ent_o_num.get()}.npy').astype(np.float64)
            fo_img_origin = fo_img.copy()
            # 处理偏移
            f_shift = int(self.ent_f_shift.get())
            o_shift = int(self.ent_o_shift.get())
            f_img = shift_image(f_img, f_shift)
            o_img = shift_image(o_img, o_shift)
            # 去除<1%最大值的噪点
            denoise_thres = float(self.ent_denoise.get())
            mask = (f_img < (np.max(f_img)*denoise_thres)) & (o_img < (np.max(o_img)*denoise_thres)) & (fo_img < (np.max(fo_img)*denoise_thres))
            # mask = fo_img < (np.max(fo_img) * denoise_thres)
            f_img[mask] = 0
            o_img[mask] = 0
            fo_img[mask] = 0
            # 计算和、差值图像
            combined = f_img + o_img
            diff = combined-fo_img
            global_of = float(self.ent_globalof.get())
            global_mix_ratio = global_of / (1 + global_of)
            # 计算掩模并显示有效数据
            coords_me = generate_mask_coords(int(self.ent_cx.get()), int(self.ent_cy.get()), int(self.ent_major.get()), int(self.ent_minor.get()), f_img.shape)
            # 计算流量轮廓
            thres_contour = float(self.ent_thres_contour.get())
            contour_value, boundary_points = 0, []
            if self.show_contour:
                contour_value, boundary_points = calc_flowrate_contour(fo_img_origin, thres_contour)
            # 计算流量中心线
            Up_limit = int(self.ent_thres_Q_center_up.get())
            Down_limit = int(self.ent_thres_Q_center_down.get())
            max_k = int(self.ent_max_k.get())
            a, c, r_squared, origin_point, fitted_point = calc_Q_center_line(fo_img_origin, Up_limit, Down_limit, max_k, int(self.ent_cx.get()), int(self.ent_cy.get()), int(self.ent_Q_center_shift.get()))
            # 计算混合效率参数
            abs_val, me1, me2, me3 = calc_result(f_img, o_img, fo_img, coords_me, global_mix_ratio)
            self.lbl_res1.config(text=f'f_shift: {f_shift}; o_shift: {o_shift}; 差绝对值和{abs_val:.2f}')
            self.lbl_res2.config(text=f'O-F混合效率: {me3:.4f}; F-FO混合效率: {me2:.4f}; O-FO混合效率: {me1:.4f}; 平均混合效率:{(me1+me2+me3)/3:.4f}; 流量轮廓值:{contour_value}')
            self.lbl_res3.config(text=f'流量中心线方程：y={a:.4f}x^2+{c:.4f}; R2={r_squared:.4f}')

            # 计算图像
            mix_ratio_o_f = np.where((f_img > 0) | (o_img > 0), (o_img / (o_img + global_of * f_img) * 254 + 1).astype(np.uint8), 0) # 归一化方式为(o/f)/(o/f+O/F)，上下同乘f
            _, mix_efficiency_o_f = calc_mixing_efficiency(o_img, combined, global_mix_ratio)

            mix_ratio_f_fo = np.where((f_img > 0) | ((fo_img - f_img) > 0), ((fo_img - f_img) / ((fo_img - f_img) + global_of * f_img) * 254 + 1).astype(np.uint8), 0)
            mix_ratio_f_fo[fo_img - f_img <0] = 1 #此时f_img一定>0，视为o=0，所以设置1
            _, mix_efficiency_f_fo = calc_mixing_efficiency(f_img, fo_img, global_mix_ratio)

            mix_ratio_o_fo = np.where(((fo_img - o_img) > 0) | (o_img > 0), (o_img / (o_img + global_of * (fo_img - o_img)) * 254 + 1).astype(np.uint8), 0)
            mix_ratio_o_fo[fo_img - o_img <0] = 255 #此时o_img一定>0，视为f=0，所以设置255
            _, mix_efficiency_o_fo = calc_mixing_efficiency(o_img, fo_img, global_mix_ratio)

            mx = max(mix_efficiency_o_f.max(), mix_efficiency_f_fo.max(), mix_efficiency_o_fo.max())
            mix_efficiency_o_f = np.array(mix_efficiency_o_f / mx * 255).astype(np.uint8)
            mix_efficiency_f_fo = np.array(mix_efficiency_f_fo / mx * 255).astype(np.uint8)
            mix_efficiency_o_fo = np.array(mix_efficiency_o_fo / mx * 255).astype(np.uint8)

            # 更新所有图像
            gain = float(self.ent_gain.get())
            zoom = float(self.ent_zoom.get())
            crosshair = (int(self.ent_cx.get()), int(self.ent_cy.get()), int(self.ent_line_width.get())) if self.crosshair else None
            ellipse = (int(self.ent_cx.get()), int(self.ent_cy.get()), int(self.ent_major.get()), int(self.ent_minor.get()), int(self.ent_line_width.get())) if self.show_ellipse else None
            denoise = float(self.ent_denoise.get())
            save_name = self.ent_save_name.get()
            for idx, img in enumerate([f_img, o_img, fo_img, diff, -diff]):
                if self.show_contour:
                    cache = plot_black_points(img, boundary_points)
                elif self.show_Q_center:
                    cache = img[:]
                    cache[origin_point[1], origin_point[0]] = 0
                elif self.show_Q_center_fitted:
                    cache = img[:]
                    cache[fitted_point[1], fitted_point[0]] = 0
                else:
                    cache = img
                self.list_display[idx].update(cache, gain, BlackedJet_colormap(), zoom, crosshair, ellipse, denoise, save_name)
            for idx, img in enumerate([mix_ratio_o_f, mix_ratio_f_fo, mix_ratio_o_fo]):
                if self.show_contour:
                    cache = plot_black_points(img, boundary_points)
                else:
                    cache = img
                self.list_display[idx+5].update(cache, 1, BlueWhiteRed_colormap(), zoom, crosshair, ellipse, denoise, save_name)
            for idx, img in enumerate([mix_efficiency_o_f, mix_efficiency_f_fo, mix_efficiency_o_fo]):
                if self.show_contour:
                    cache = plot_black_points(img, boundary_points)
                else:
                    cache = img
                self.list_display[idx+8].update(cache, 1, BlackedJet_colormap(), zoom, crosshair, ellipse, denoise, save_name)
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

    def toggle_contour(self):
        self.show_contour = not self.show_contour
        self.btn_contour.config(text="关闭轮廓" if self.show_contour else "显示轮廓")
        self.process()

    def toggle_Q_center(self):
        self.show_Q_center = not self.show_Q_center
        self.btn_Q_center.config(text="关闭流量中心线" if self.show_Q_center else "显示流量中心线")
        self.process()

    def toggle_Q_center_fitted(self):
        self.show_Q_center_fitted = not self.show_Q_center_fitted
        self.btn_Q_center_fitted.config(text="关闭拟合流量中心线" if self.show_Q_center_fitted else "显示拟合流量中心线")
        self.process()

def on_close():
    app.ConfigManager.save_config()
    root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root, '6.1')
    app.pack()
    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()