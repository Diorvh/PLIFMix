import tkinter as tk
from tkinter import ttk, filedialog
import cv2,os,re
import numpy as np
from PIL import Image, ImageTk
from lib.lib_ConfigManager import ConfigManager
from lib.lib_cv import *
from lib.lib_5 import *
from lib.lib_61 import *
from lib.lib_excel import *
PIXEL_SIZE = 0.14 # mm/pixel
DIAMETER_FUEL = 0.5025 # mm

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
        self.show_crosshair = True
        self.show_ellipse = True
        self.LINE_WIDTH=5
        self.show_Q_center = 'nonshow'
        self.show_img = 'ori_img'

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
        ttk.Label(frm1, text="增益为0则为每张图按最大值归一化后保存").pack(side=tk.LEFT)

        for ent in ['ent_globalof', 'ent_zoom', 'ent_gain']:
            getattr(self, ent).bind('<Return>', self.process)

        # 第二行：中心点和椭圆控制
        frm2 = frm1
        # frm2.pack(fill=tk.X)

        ttk.Label(frm2, text="中心点坐标：").pack(side=tk.LEFT)
        self.ent_cxf = ttk.Entry(frm2, width=5)
        self.ent_cxf.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('center_xf', self.ent_cxf)
        self.ent_cyf = ttk.Entry(frm2, width=5)
        self.ent_cyf.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('center_yf', self.ent_cyf)
        self.ent_cxo = ttk.Entry(frm2, width=5)
        self.ent_cxo.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('center_xo', self.ent_cxo)
        self.ent_cyo = ttk.Entry(frm2, width=5)
        self.ent_cyo.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('center_yo', self.ent_cyo)
        self.ent_cxfo = ttk.Entry(frm2, width=5)
        self.ent_cxfo.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('center_xfo', self.ent_cxfo)
        self.ent_cyfo = ttk.Entry(frm2, width=5)
        self.ent_cyfo.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('center_yfo', self.ent_cyfo)

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

        for ent in ['ent_major', 'ent_minor', 'ent_line_width', 'ent_cxf', 'ent_cyf', 'ent_cxo', 'ent_cyo', 'ent_cxfo', 'ent_cyfo']:
            getattr(self, ent).bind('<Return>', self.process)
        

        # 第三行：平移量
        frm3 = frm1
        # frm3.pack(fill=tk.X)

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

        # 第五行：轮廓与中心线控制
        frm5 = ttk.Frame(self)
        frm5.pack(fill=tk.X)
        ttk.Label(frm5, text="轮廓流量比例").pack(side=tk.LEFT)
        self.ent_thres_contour = ttk.Entry(frm5, width=5)
        self.ent_thres_contour.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('thres_contour', self.ent_thres_contour)
        
        ttk.Label(frm5, text="流量中心线上下范围").pack(side=tk.LEFT)
        self.ent_thres_Q_center_up = ttk.Entry(frm5, width=5)
        self.ent_thres_Q_center_up.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('thres_Q_center_up', self.ent_thres_Q_center_up)
        self.ent_thres_Q_center_down = ttk.Entry(frm5, width=5)
        self.ent_thres_Q_center_down.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('thres_Q_center_down', self.ent_thres_Q_center_down)
        ttk.Label(frm5, text="最大值个数").pack(side=tk.LEFT)
        self.ent_max_k = ttk.Entry(frm5, width=5)
        self.ent_max_k.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('max_k', self.ent_max_k)
        ttk.Label(frm5, text="图像顺时针旋转角度后拟合：").pack(side=tk.LEFT)
        self.ent_rotate_anglef = ttk.Entry(frm5, width=5)
        self.ent_rotate_anglef.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('rotate_anglef', self.ent_rotate_anglef)
        self.ent_rotate_angleo = ttk.Entry(frm5, width=5)
        self.ent_rotate_angleo.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('rotate_angleo', self.ent_rotate_angleo)
        self.ent_rotate_anglefo = ttk.Entry(frm5, width=5)
        self.ent_rotate_anglefo.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('rotate_anglefo', self.ent_rotate_anglefo)

        self.btn_contour = ttk.Button(frm5, text="切换至显示流量轮廓", command=self.toggle_contour)
        self.btn_contour.pack(side=tk.LEFT)
        self.btn_Q_center = ttk.Button(frm5, text="切换至显示流量中心线", command=self.toggle_Q_center)
        self.btn_Q_center.pack(side=tk.LEFT)
        self.btn_Q_center_fitted = ttk.Button(frm5, text="切换至显示拟合流量中心线", command=self.toggle_Q_center_fitted)
        self.btn_Q_center_fitted.pack(side=tk.LEFT)
        self.btn_Q_center_nonshow = ttk.Button(frm5, text="切换至不显示", command=self.toggle_Q_center_nonshow)
        self.btn_Q_center_nonshow.pack(side=tk.LEFT)
        self.btn_bent_img = ttk.Button(frm5, text="切换至显示掰直后图像", command=self.toggle_bent_img)
        self.btn_bent_img.pack(side=tk.LEFT)
        self.btn_fit_img = ttk.Button(frm5, text="切换至显示拟合图像", command=self.toggle_fit_img)
        self.btn_fit_img.pack(side=tk.LEFT)
        self.btn_delta_img = ttk.Button(frm5, text="切换至显示拟合图像-掰直后图像", command=self.toggle_delta_img)
        self.btn_delta_img.pack(side=tk.LEFT)
        self.btn_ori_img = ttk.Button(frm5, text="切换至显示原始图像", command=self.toggle_ori_img)
        self.btn_ori_img.pack(side=tk.LEFT)
        
        # 第六行：曲线拟合结果
        frm6 = ttk.Frame(self)
        frm6.pack(fill=tk.X)
        self.lbl_res2 = ttk.Label(frm6,text='计算结果')
        self.lbl_res2.pack(anchor='w')

        # 第七行：曲面拟合结果
        frm7 = ttk.Frame(self)
        frm7.pack(fill=tk.X)
        self.lbl_res3 = ttk.Label(frm7,text='计算结果')
        self.lbl_res3.pack(anchor='w')


        for ent in ['ent_thres_contour', 'ent_thres_Q_center_up', 'ent_thres_Q_center_down', 'ent_max_k', 'ent_rotate_anglef', 'ent_rotate_angleo', 'ent_rotate_anglefo']:
            getattr(self, ent).bind('<Return>', self.process)

    def setup_image_displays(self):
        # 第五行：图像显示
        frm5 = ttk.Frame(self)
        frm5.pack(fill=tk.X)
        
        self.list_display=[]
        self.img_display=[]
        display_title = ['f', 'o', 'fo', 'f+o-fo', 'fo-f-o', 'oDf+o', 'fDfo','oDfo', 'ME_oDf+o', 'ME_fDfo','ME_oDfo','mix_curve']
        display_position = [(0,0),(0,1),(0,2),(0,3),(1,3),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(2,3)]
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
            path = os.path.dirname(self.base_folder.get())
            # 处理偏移
            f_shift = int(self.ent_f_shift.get())
            o_shift = int(self.ent_o_shift.get())
            f_img = shift_image(f_img, f_shift)
            o_img = shift_image(o_img, o_shift)
            # 去除<1%最大值的噪点
            denoise_thres = float(self.ent_denoise.get())
            mask = (f_img < (np.max(f_img)*denoise_thres)) & (o_img < (np.max(o_img)*denoise_thres))# & (fo_img < (np.max(fo_img)*denoise_thres))
            # mask = fo_img < (np.max(fo_img) * denoise_thres)
            f_img[mask] = 0
            o_img[mask] = 0
            fo_img[mask] = 0
            # 计算和、差值图像
            combined = f_img + o_img
            diff = combined-fo_img
            global_of = float(self.ent_globalof.get())
            global_mix_ratio = global_of / (1 + global_of)
            record_OF = np.sum(o_img)/np.sum(f_img)
            o_img = o_img * global_of / record_OF
            
            print(np.sum(f_img), np.sum(o_img), np.sum(fo_img), record_OF)
            # 计算掩模并显示有效数据
            coords_me = generate_mask_coords(int(self.ent_cxfo.get()), int(self.ent_cyfo.get()), int(self.ent_major.get()), int(self.ent_minor.get()), f_img.shape)
            # 计算混合效率参数
            abs_val, me1, me2, me3 = calc_result(f_img, o_img, fo_img, coords_me, global_mix_ratio)
            self.lbl_res1.config(text=f'差绝对值和{abs_val:.2f}; O-F混合效率: {me3:.4f}; F-FO混合效率: {me2:.4f}; O-FO混合效率: {me1:.4f}; 平均混合效率:{(me1+me2+me3)/3:.4f}; 实拍OF:{record_OF:.4f}')
            
            # 更新所有图像所需的变量
            gain = float(self.ent_gain.get())
            zoom = float(self.ent_zoom.get())
            crosshair = (int(self.ent_cxfo.get()), int(self.ent_cyfo.get()), int(self.ent_line_width.get())) if self.show_crosshair else None
            ellipse = (int(self.ent_cxfo.get()), int(self.ent_cyfo.get()), int(self.ent_major.get()), int(self.ent_minor.get()), int(self.ent_line_width.get())) if self.show_ellipse else None
            denoise = float(self.ent_denoise.get())
            save_name = self.ent_save_name.get()
            thres_contour = float(self.ent_thres_contour.get())
            H, W = f_img.shape

            Up_limit = int(self.ent_thres_Q_center_up.get())
            Down_limit = int(self.ent_thres_Q_center_down.get())
            max_k = int(self.ent_max_k.get())
            lbl2 = ''
            lbl3 = ''
            name_list = ['f_img', 'o_img', 'fo_img']
            center = [(int(self.ent_cxf.get()), int(self.ent_cyf.get())), (int(self.ent_cxo.get()), int(self.ent_cyo.get())), (int(self.ent_cxfo.get()), int(self.ent_cyfo.get()))]
            rotate_angle = [float(self.ent_rotate_anglef.get()), float(self.ent_rotate_angleo.get()), float(self.ent_rotate_anglefo.get())]
            wb = openpyxl.Workbook()
            ws = wb.active
            for num, img in enumerate([f_img, o_img]):
                # 计算流量轮廓
                cx, cy = center[num]
                contour_value, boundary_points = 0, []
                if self.show_Q_center == 'contour':
                    contour_value, boundary_points = calc_flowrate_contour(img, thres_contour)
                # 计算流量中心线
                a, b, r_squared, origin_point, fitted_point, verti_px_rotated = fit_center_line(img, Up_limit, Down_limit, max_k, cx, cy, rotate_angle[num])
                lbl2 += f'{name_list[num]}流量轮廓值:{contour_value};流量中心线方程：y={a:.4f}*sqrt(x^2+{b*b:.4f})-{a:.4f}*{b:.4f}; R2={r_squared:.4f}     '

                # 图像概率密度归一化
                NORM_COEFF = np.sum(img)
                norm_img = img / NORM_COEFF # 防止出现数值问题

                # 计算横向分布
                hori_px = np.array(range(-200, 200))# np.array(range(img.shape[1]))-cx
                hori_py = np.zeros_like(hori_px)
                hori_px_rotated, hori_py_rotated = rotate_points_clockwise(hori_px, hori_py, -rotate_angle[num])
                hori_intensity = norm_img[np.round(-hori_py_rotated + cy).astype(np.int32), np.round(hori_px_rotated + cx).astype(np.int32)]
                def func_fit(x, a, b, c):
                    return a / (x ** 2 * b ** 2 + 1) ** c
                ah, bh, ch, r_squaredh, y_pred = func_fit_3(hori_px * PIXEL_SIZE / DIAMETER_FUEL, hori_intensity, func_fit)
                plot_scatter_and_line(hori_px, hori_intensity, hori_px, y_pred, path+'\\横向分布'+f'_{name_list[num]}.png')
                lbl2 += f'{name_list[num]}横向强度分布拟合：y={ah*1e4:.4f}E-4/(x^2*{bh:.4f}^2+1)^{ch:.4f}; R2={r_squaredh:.4f}     '
                # 计算纵向分布
                verti_py, verti_px = origin_point
                verti_intensity = norm_img[verti_py, verti_px]
                av, bv, cv, r_squaredv, y_pred = func_fit_3(verti_px_rotated * PIXEL_SIZE / DIAMETER_FUEL, verti_intensity, func_fit)
                plot_scatter_and_line(verti_px_rotated, verti_intensity, verti_px_rotated, y_pred, path+'\\纵向分布'+f'_{name_list[num]}.png')
                lbl2 += f'{name_list[num]}纵向强度分布拟合：y={av*1e4:.4f}E-4/(x^2*{bv:.4f}^2+1)^{cv:.4f}; R2={r_squaredv:.4f}\n'
                # 掰直图像
                bent_img = np.zeros_like(img)
                b_cy, b_cx = origin_point
                x_coords = np.arange(W)
                for i, y in enumerate(b_cy):
                    src_x = x_coords - (cx - b_cx[i])
                    mask = (0 <= src_x) & (src_x < W)
                    bent_img[y, x_coords[mask]] = norm_img[y, src_x[mask]]
                # 优化RMSE以拟合相关系数rou
                rho_limit = (0.01, 100)
                def predict_img_func(rho):
                     # 创建坐标网格
                    X, Y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
                    trans_x = (X - cx) * PIXEL_SIZE / DIAMETER_FUEL
                    trans_y = (Y - cy) * PIXEL_SIZE / DIAMETER_FUEL
                    a = (ah + av) / 2
                    predict_img = a / ( ((trans_x * trans_x * bh * bh +1) ** ch - 1) +
                                        ((trans_y * trans_y * bv * bv +1) ** cv - 1) +
                                        # (rho ** (np.abs(trans_x * bh) ** (1/ch) * np.abs(trans_y * bv) ** (1/cv))-1) + 1)
                                        (np.abs(trans_x * bh) ** (rho) * np.abs(trans_y * bv) ** (rho)) + 1)
                                        # ((np.abs(trans_x * bh * trans_y * bv) +1) ** rho - 1) + 1)
                    predict_img = predict_img / np.sum(predict_img)
                    return predict_img
                def object_loss(rho):
                    ss_res = np.sum((bent_img - predict_img_func(rho)) ** 2)
                    return ss_res
                if self.show_img == 'fit_img' or self.show_img == 'delta_img':
                    try :
                        from scipy import optimize
                        res = optimize.minimize_scalar(object_loss, bounds = rho_limit, method = 'bounded')
                        ss_tot = np.sum((bent_img - np.mean(bent_img)) ** 2)
                        best_rho = res.x
                        r_squaredp = 1 - (res.fun / ss_tot) if ss_tot != 0 else 0
                    except Exception as e:
                        best_rho = 0
                        r_squaredp = 0
                        print(f"Optimization error for {name_list[num]}: {e}")
                else:
                    best_rho = 0
                    r_squaredp = 0
                lbl3 += f'{name_list[num]}横纵向联合分布拟合：rho={best_rho:.4f}; R2={r_squaredp:.4f}\n'

                if self.show_Q_center == 'contour':
                    special_points = list(zip(*boundary_points))
                elif self.show_Q_center == 'Q_center':
                    special_points = origin_point
                elif self.show_Q_center == 'Q_center_fitted':
                    special_points = fitted_point
                else:
                    special_points = []
                show_img = norm_img*NORM_COEFF
                if self.show_img == 'bent_img':
                    show_img = bent_img*NORM_COEFF
                elif self.show_img == 'fit_img':
                    show_img = predict_img_func(best_rho)*NORM_COEFF
                elif self.show_img == 'delta_img':
                    show_img = np.abs(predict_img_func(best_rho) - bent_img)*NORM_COEFF
                else:
                    show_img = norm_img * NORM_COEFF
                self.list_display[num].update(show_img, gain, BlackedJet_colormap(), zoom, (cx, cy, int(self.ent_line_width.get())) if self.show_crosshair else None, ellipse, denoise, save_name, special_points)
                write_flush = [cx, cy, rotate_angle[num], a, b, r_squared, ah, bh, ch, r_squaredh, av, bv, cv, r_squaredv, best_rho, r_squaredp]
                for j, str in enumerate(write_flush):
                    ws.cell(row=num*2+1, column=j+1, value=str)
                # print(f'{name_list[num]}:\n{ah*1e4:.4f}\n{ch:.4f}\n{r_squaredh:.4f}\n{av*1e4:.4f}\n{bv:.4f}\n{cv:.4f}\n{r_squaredv:.4f}\n')
            try:
                file_path = path+'\\cache.xlsx'
                wb.save(file_path)
                os.system(f'start excel "{file_path}"')
            except Exception as e:
                print(f"Error saving or opening Excel file: {e}")
            self.lbl_res2.config(text=lbl2)
            self.lbl_res3.config(text=lbl3)
            cx, cy = center[2]
            self.list_display[2].update(fo_img, gain, BlackedJet_colormap(), zoom, (cx, cy, int(self.ent_line_width.get())) if self.show_crosshair else None, ellipse, denoise, save_name, [])

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

            mix_ratio_o_f_origin = np.where((f_img > 0) | (o_img > 0), (o_img / (o_img + global_of * f_img))+1, 0)
            row = int(self.ent_cyfo.get())
            mix_curve_x = np.array(range(f_img.shape[1]))
            mix_curve_y = mix_ratio_o_f_origin[row, :]
            mask = (mix_curve_y != 0)
            mix_curve_x = mix_curve_x[mask]
            mix_curve_y = mix_curve_y[mask]
            mix_curve_y = mix_curve_y - 1  # 恢复到0-1范围
            mix_curve_img = plot_data_advanced(mix_curve_x, mix_curve_y, title = '混合分数曲线y=0', xlabel='x', ylabel='混合分数', axislimit=[(500,1000),(0.2,0.9)])
            write_excel_func(mix_curve_x, mix_curve_y, r'D:\工作文件同步\20250826孔径比影响'+f'\\{save_name}混合分数曲线.xlsx')
            # 更新剩余图像
            for idx, img in enumerate([diff, -diff]):
                self.list_display[idx+3].update(img, gain, BlackedJet_colormap(), zoom, crosshair, ellipse, denoise, save_name, [])
            for idx, img in enumerate([mix_ratio_o_f, mix_ratio_f_fo, mix_ratio_o_fo]):
                self.list_display[idx+5].update(img, 1, BlueWhiteRed_colormap(), zoom, crosshair, ellipse, denoise, save_name, [])
            for idx, img in enumerate([mix_efficiency_o_f, mix_efficiency_f_fo, mix_efficiency_o_fo]):
                self.list_display[idx+8].update(img, 1, BlackedJet_colormap(), zoom, crosshair, ellipse, denoise, save_name, [])
            self.list_display[-1].update(mix_curve_img, zoom = zoom, denoise = denoise, save_name_suffix = save_name)
        except Exception as e:
            print(e)

    # 伪彩图切换功能
    def toggle_pseudo(self):
        self.is_pseudo = not self.is_pseudo
        self.btn_pseudo.config(text="关闭伪彩图" if self.is_pseudo else "显示伪彩图")
        self.process()

    # 十字线控制
    def toggle_crosshair(self):
        self.show_crosshair = not self.show_crosshair
        self.btn_cross.config(text="关闭十字线" if self.show_crosshair else "显示十字线")
        self.process()

    # 椭圆控制
    def toggle_ellipse(self):
        self.show_ellipse = not self.show_ellipse
        self.btn_ellipse.config(text="关闭椭圆" if self.show_ellipse else "显示椭圆")
        self.process()

    def toggle_contour(self):
        self.show_Q_center = 'contour'
        self.process()

    def toggle_Q_center(self):
        self.show_Q_center = 'Q_center'
        self.process()

    def toggle_Q_center_fitted(self):
        self.show_Q_center = 'Q_center_fitted'
        self.process()

    def toggle_Q_center_nonshow(self):
        self.show_Q_center = 'nonshow'
        self.process()
        
    def toggle_bent_img(self):
        self.show_img = 'bent_img'
        self.process()

    def toggle_fit_img(self):
        self.show_img = 'fit_img'
        self.process()

    def toggle_delta_img(self):
        self.show_img = 'delta_img'
        self.process()

    def toggle_ori_img(self):
        self.show_img = 'ori_img'
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