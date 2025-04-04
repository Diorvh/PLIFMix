from calendar import c
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize
from lib.lib_cv import *

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ImageAnalysisApp(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.folder_path = tk.StringVar()
        self.click_coord = None
        self.all_images_data = None
        self.error_map = None
        self.k_matrix = None
        self.current_vmax = None
        self.radius = 0
        self.circle_id = None  # 用于存储绘制的圆形ID
        
        # 初始化界面组件
        self.create_widgets()

    def create_widgets(self):
        """创建所有界面组件"""
        # 顶部路径选择
        top_frame = ttk.Frame(self)
        top_frame.grid(row=0, column=0, columnspan=2)

        ttk.Label(top_frame, text="图片文件夹路径：").pack(side=tk.LEFT)
        ttk.Label(top_frame, textvariable=self.folder_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(top_frame, text="选择文件夹", command=self.select_folder).pack(side=tk.LEFT)
        ttk.Button(top_frame, text="保存修正矩阵", command=self.save_correct_matrix).pack(side=tk.LEFT)

        # 左侧图像显示面板
        self.left_frame = ttk.Frame(self)
        self.left_frame.grid(row=1, column=0,sticky='nsew')
        """创建左侧图像显示组件"""
        self.img_canvas = tk.Canvas(self.left_frame, bg='#333')
        self.img_canvas.pack(fill=tk.BOTH, expand=True)

        self.pixel_info = ttk.Label(self.left_frame)
        self.pixel_info.pack(anchor='w')

        self.img_canvas.bind("<Motion>", self.update_pixel_info)
        self.img_canvas.bind("<Button-1>", self.handle_image_click)

        # 右侧分析面板
        self.right_frame = ttk.Frame(self)
        self.right_frame.grid(row=1, column=1,sticky='nsew')
        """创建右侧分析面板"""
        # 热力图区域
        self.heatmap_frame = ttk.Labelframe(self.right_frame, text="全局误差分布")
        self.heatmap_frame.pack(fill=tk.BOTH, expand=True)
        self.create_heatmap_controls()

        # 曲线图区域
        self.curve_frame = ttk.Labelframe(self.right_frame, text="单点分析")
        self.curve_frame.pack(fill=tk.BOTH, expand=True)

        self.figure_curve = Figure(figsize=(8, 3))
        self.ax_curve = self.figure_curve.add_subplot(111)
        self.canvas_curve = FigureCanvasTkAgg(self.figure_curve, master=self.curve_frame)
        self.canvas_curve.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # 统计信息
        self.stats_var = tk.StringVar()
        ttk.Label(self.curve_frame, textvariable=self.stats_var).pack()

    def create_heatmap_controls(self):
        """创建热力图控制组件"""
        control_frame = ttk.Frame(self.heatmap_frame)
        control_frame.pack(fill=tk.X)

        # 颜色范围控制行
        color_frame = ttk.Frame(control_frame)
        color_frame.pack(fill=tk.X)

        ttk.Label(color_frame, text="颜色上限:").pack(side=tk.LEFT)

        self.vmax_entry = ttk.Entry(color_frame, width=10)
        self.vmax_entry.pack(side=tk.LEFT)
        self.vmax_entry.bind("<Return>", self.update_heatmap_range)

        self.auto_vmax_label = ttk.Label(color_frame, text="(自动)")
        self.auto_vmax_label.pack(side=tk.LEFT)

        # 新增的区域分析行
        analysis_frame = ttk.Frame(control_frame)
        analysis_frame.pack(fill=tk.X)

        ttk.Label(analysis_frame, text="分析半径:").pack(side=tk.LEFT)

        self.radius_entry = ttk.Entry(analysis_frame, width=10)
        self.radius_entry.pack(side=tk.LEFT)
        self.radius_entry.bind("<Return>", self.calculate_region_error)

        self.result_label = ttk.Label(analysis_frame, text="区域平均误差: -")
        self.result_label.pack(side=tk.LEFT)

        #热力图创建
        self.figure_heatmap = Figure(figsize=(8, 4))
        gs = GridSpec(1,2,width_ratios=[10,1])
        self.ax_heatmap = self.figure_heatmap.add_subplot(gs[0])
        self.cax_heatmap = self.figure_heatmap.add_subplot(gs[1])
        self.canvas_heatmap = FigureCanvasTkAgg(self.figure_heatmap, master=self.heatmap_frame)
        self.canvas_heatmap.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def select_folder(self):
        """处理文件夹选择"""
        path = filedialog.askdirectory()
        if path:
            self.folder_path.set(path)
            self.load_all_images(path)
            self.load_display_image()
            self.process_all_pixels()
    
    def save_correct_matrix(self):
        '''保存相增强器修正矩阵'''
        assert self.k_matrix is not None
        #归一化
        norm_k_matrix = self.k_matrix/self.k_matrix.max()
        correct_matrix = np.where (norm_k_matrix > 0, 1/norm_k_matrix, 0)
        folder = self.folder_path.get()
        imsave_float(correct_matrix, os.path.dirname(folder), os.path.basename(folder))
                
    def load_all_images(self, folder):
        """加载所有图像数据"""
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]            
        numeric_files = []
        for f in files:
            num = int(os.path.splitext(f)[0])
            numeric_files.append((num, f))            
        sorted_files = sorted(numeric_files, key=lambda x: x[0])
        file_paths = [os.path.join(folder, f[1]) for f in sorted_files]        
        img_stack = []
        for fp in file_paths:
            img_stack.append(cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_ANYDEPTH))
        self.all_images_data = np.dstack(img_stack)
            
    def load_display_image(self):
        """加载显示图像"""
        assert self.all_images_data is not None
        img_8bit = self.all_images_data[:,:,-1] // 256
        img_8bit = img_8bit.astype(np.uint8)
        pil_image = Image.fromarray(img_8bit)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.img_canvas.config(width=pil_image.width,height=pil_image.height)
        self.img_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        # 刷新平均误差
        # if self.radius_entry.get():
        #     self.calculate_region_error()

    def process_all_pixels(self):
        """每个点最小二乘法拟合，计算kk_matrix和最大相对误差矩阵error_map"""
        num_images = self.all_images_data.shape[2] # type: ignore
        x_data = np.arange(1, num_images+1)
        
        X = x_data.reshape(-1, 1)
        XtX_inv = np.linalg.inv(X.T @ X)
        sum_xy = np.tensordot(self.all_images_data, x_data, axes=([2],[0]))# type: ignore
        
        self.k_matrix = XtX_inv[0,0] * sum_xy
        # 清零边界
        self.k_matrix[self.k_matrix < 1e3] = 0
        
        y_pred = self.k_matrix[..., None] * x_data
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_errors = np.abs((y_pred - self.all_images_data) / self.all_images_data)
        rel_errors[np.isinf(rel_errors)] = 0
        self.error_map = np.max(rel_errors, axis=2)
        
        self.current_vmax = self.error_map.max()
        self.update_heatmap()

    def update_heatmap(self):
        '''创建或更新热力图'''
        assert self.error_map is not None
        if not hasattr(self, 'im'):  # 首次运行
            # 创建主热力图
            self.im = self.ax_heatmap.imshow(self.error_map, cmap='viridis',
                vmin=0,vmax=self.current_vmax,
                origin='lower',aspect='auto',interpolation='nearest'  # 保持像素化显示
            )
            # 创建colorbar并固定格式
            self.cbar = self.figure_heatmap.colorbar(self.im, cax=self.cax_heatmap)
            self.cbar.set_label('最大相对误差', rotation=270, labelpad=15)
            # 初始化坐标轴标签
            self.ax_heatmap.set_xlabel("X Coordinate")
            self.ax_heatmap.set_ylabel("Y Coordinate")
            self.ax_heatmap.set_title(f"Heatmap (0-{self.current_vmax})")
        else:  # 后续更新
            # 更新图像数据
            self.im.set_data(self.error_map)
            self.im.set_clim(0, self.current_vmax)  # 动态设置颜色范围
            self.ax_heatmap.set_title(f"Heatmap (0-{self.current_vmax})")
            # 自动调整colorbar刻度
            self.cbar.update_normal(self.im)  # 关键！同步colorbar范围
        # 更新控件显示
        self.auto_vmax_label.config(text=f"(自动: {np.nanmax(self.error_map):.4f})")
        self.vmax_entry.delete(0, tk.END)
        self.vmax_entry.insert(0, f"{self.current_vmax:.4f}")
        # 智能布局调整（仅在必要时）
        if not hasattr(self, '_laid_out'):
            self.figure_heatmap.tight_layout()
            self._laid_out = True  # 标记已完成初始布局
        # 异步渲染优化
        self.canvas_heatmap.draw_idle()

    def update_pixel_info(self, event):
        """更新像素信息"""
        if self.all_images_data is None:
            return
        x, y = event.x, event.y
        if 0 <= x < self.all_images_data.shape[1] and 0 <= y < self.all_images_data.shape[0]:
            value = self.all_images_data[y, x, -1]
            self.pixel_info.config(text=f"坐标: ({x}, {y})  当前值: {int(value)}")

    def handle_image_click(self, event):
        """处理图像点击事件"""
        if self.all_images_data is None:
            return
        x, y = event.x, event.y
        if 0 <= x < self.all_images_data.shape[1] and 0 <= y < self.all_images_data.shape[0]:
            self.click_coord = (y, x)
            self.update_curve()
    
    def update_curve(self):
        """更新曲线分析图"""
        assert self.click_coord is not None
        assert self.all_images_data is not None
        assert self.k_matrix is not None
        assert self.error_map is not None

        self.ax_curve.clear()
        y, x = self.click_coord
        num_images = self.all_images_data.shape[2]
        x_data = np.arange(1, num_images+1)
        y_values = self.all_images_data[y, x, :]
        k = self.k_matrix[y, x]
        
        y_pred = k * x_data
        ss_res = np.sum((y_values - y_pred)**2)
        ss_tot = np.sum((y_values - np.mean(y_values))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        error = (y_pred - y_values) / (y_values)
        # error[:4] = 0
        max_val, idx = error.max(), np.argmax(error)
        # print(max_val, self.error_map[y, x])
        
        stats_text = (f"拟合系数 k: {k:.4e} | R²: {r_squared:.4f} | 最大相对误差: {max_val:.2%} | 最大误差位置: {x_data[idx]}")
        self.stats_var.set(stats_text)
        
        self.ax_curve.plot(x_data, y_values, 'b.-', markersize=5, label='原始数据')
        self.ax_curve.plot(x_data, y_pred, 'r-', linewidth=1.5, label='拟合结果')
        self.ax_curve.set_xlabel("图像序号")
        self.ax_curve.set_ylabel("强度值")
        self.ax_curve.set_title(f"坐标 ({x}, {y}) 强度变化分析")
        self.ax_curve.legend()
        self.ax_curve.grid(True, alpha=0.3)
        self.canvas_curve.draw_idle()

    def calculate_region_error(self, event=None):
        """计算圆形区域内的平均误差"""
        assert self.error_map is not None
        # 获取半径
        r = int(self.radius_entry.get())
        h, w = self.error_map.shape
        center_y, center_x = h // 2, w // 2
        # 生成坐标网格
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= r**2
        # 计算平均误差
        valid_errors = self.error_map[mask]        
        avg_error = np.nanmean(valid_errors)
        self.result_label.config(text=f"区域平均误差: {avg_error:.4f}")
        # 更新绘制的圆形
        self.draw_circle(r)

    def draw_circle(self, radius):
        """在左侧图像绘制圆形"""
        assert self.all_images_data is not None
        # 删除旧圆形
        if self.circle_id:
            self.img_canvas.delete(self.circle_id)
        # 获取图像尺寸
        img_h, img_w = self.all_images_data.shape[:2]
        canvas_w = self.img_canvas.winfo_width()
        canvas_h = self.img_canvas.winfo_height()
        # 计算缩放比例
        scale_x = canvas_w / img_w
        scale_y = canvas_h / img_h
        scale = min(scale_x, scale_y)
        # 计算绘制参数
        center_x = canvas_w / 2
        center_y = canvas_h / 2
        radius_pixels = radius * scale
        # 绘制圆形（确保保持圆形形状）
        self.circle_id = self.img_canvas.create_oval(
            center_x - radius_pixels, center_y - radius_pixels,
            center_x + radius_pixels, center_y + radius_pixels,
            outline='red', width=2
        )

    def update_heatmap_range(self, event=None):
        """处理颜色范围更新"""
        assert self.error_map is not None
        try:
            input_text = self.vmax_entry.get().strip()
            if input_text == "":
                self.current_vmax = np.nanmax(self.error_map)
            else:
                self.current_vmax = float(input_text)
            self.update_heatmap()
        except ValueError as e:
            messagebox.showerror("输入错误", f"无效输入: {str(e)}")
            self.vmax_entry.delete(0, tk.END)
            self.vmax_entry.insert(0, f"{self.current_vmax:.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("高级图像分析工具 - 最终稳定版")
    app = ImageAnalysisApp(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()