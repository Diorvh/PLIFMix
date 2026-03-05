import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from lib.lib_cv import *

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ImageAnalysisApp(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.folder_path = tk.StringVar()
        self.avg_img_path = tk.StringVar()
        self.click_coord = None
        self.original_16bit = None
        self.create_widgets()
        
        # 初始化绘图组件
        self.figure = plt.Figure(figsize=(6, 4)) # type: ignore
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame_right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

    def create_widgets(self):
        # Frame1 - 顶部路径选择
        frame1 = ttk.Frame(self)
        frame1.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        ttk.Label(frame1, text="图片文件夹路径：").pack(side=tk.LEFT)
        ttk.Label(frame1, textvariable=self.folder_path, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(frame1, text="选择文件夹", command=self.select_folder).pack(side=tk.RIGHT)

        # Frame2-Left
        self.frame_left = ttk.Frame(self)
        self.frame_left.grid(row=1, column=0, padx=5, pady=5)
        self.create_left_panel()

        # Frame2-Right
        self.frame_right = ttk.Frame(self)
        self.frame_right.grid(row=1, column=1, padx=5, pady=5)
        self.frame_right.columnconfigure(0, weight=1)
        self.frame_right.rowconfigure(0, weight=1)
        
        # 统计信息标签
        self.stats_var = tk.StringVar()
        ttk.Label(self.frame_right, textvariable=self.stats_var).grid(row=1, column=0, pady=5)

    def create_left_panel(self):
        # 平均图路径选择
        path_frame = ttk.Frame(self.frame_left)
        path_frame.grid(row=0, column=0, sticky='ew', pady=5)
        
        ttk.Label(path_frame, text="平均图路径：").pack(side=tk.LEFT)
        ttk.Label(path_frame, textvariable=self.avg_img_path, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="选择文件", command=self.select_avg_image).pack(side=tk.RIGHT)

        # 图片显示区域
        self.img_canvas = tk.Canvas(self.frame_left, highlightthickness=0)
        self.img_canvas.grid(row=1, column=0)

        # 像素信息显示
        self.pixel_info = ttk.Label(self.frame_left)
        self.pixel_info.grid(row=2, column=0, pady=5, sticky='w')

        # 事件绑定
        self.img_canvas.bind("<Motion>", self.update_pixel_info)
        self.img_canvas.bind("<Button-1>", self.handle_image_click)

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_path.set(path)

    def select_avg_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.avg_img_path.set(path)
            self.load_avg_image(path)

    def load_avg_image(self, path):
        try:
            # 处理中文路径
            self.original_16bit = cvimread(path)
            if self.original_16bit.ndim == 3:
                self.original_16bit = self.original_16bit[:, :, 1]
            
            # 转换为8位
            img_8bit = self.original_16bit # // 256
            img_8bit = img_8bit.astype(np.uint8)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(img_8bit)
            self.tk_image = ImageTk.PhotoImage(pil_image)
            
            # 更新显示尺寸
            self.img_canvas.config(
                width=pil_image.width,
                height=pil_image.height
            )
            self.img_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
        except Exception as e:
            self.avg_img_path.set(f"加载失败: {str(e)}")

    def update_pixel_info(self, event):
        if self.original_16bit is not None:
            x, y = event.x, event.y
            if 0 <= x < self.original_16bit.shape[1] and 0 <= y < self.original_16bit.shape[0]:
                value = self.original_16bit[y, x]
                self.pixel_info.config(text=f"坐标: ({x}, {y})  强度值: {int(value)}")

    def handle_image_click(self, event):
        if self.original_16bit is not None:
            x, y = event.x, event.y
            if 0 <= x < self.original_16bit.shape[1] and 0 <= y < self.original_16bit.shape[0]:
                self.click_coord = (y, x)
                self.update_plot()

    def read_image_data(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(img_bytes, cv2.IMREAD_ANYDEPTH)
        except Exception as e:
            print(f"读取失败: {file_path} - {str(e)}")
            return None

    def update_plot(self):
        if not self.folder_path.get() or not self.click_coord:
            return
        
        try:
            # 获取文件列表
            folder = self.folder_path.get()
            # files = sorted(
            #     [f for f in os.listdir(folder)],
            #     key=lambda x: int(x.split('.')[0])
            # )
            files = [f for f in os.listdir(folder)]
            file_paths = [os.path.join(folder, f) for f in files]
            
            # 读取所有图像数据
            y_values = []
            for fp in file_paths:
                img = cvimread(fp)
                # img = BGR2GRAY_float(img)
                if img is not None:
                    y, x = self.click_coord
                    y_values.append(img[y, x, 1])  # 取绿色通道
            
            # 计算统计指标
            avg = np.mean(y_values)
            rng = np.ptp(y_values)
            rel_rng = rng / avg if avg != 0 else 0
            std = np.std(y_values)
            rel_std = std / avg if avg != 0 else 0
            
            stats_text = (
                f"平均值: {avg:.2f} | "
                f"极差: {rng} | "
                f"相对极差: {rel_rng:.2%} | "
                f"相对标准差: {rel_std:.2%}"
            )
            self.stats_var.set(stats_text)
            
            # 更新绘图
            self.ax.clear()
            self.ax.plot(range(len(y_values)), y_values, 'b-')
            self.ax.set_xlabel("图像编号")
            self.ax.set_ylabel("强度值")
            self.ax.set_title(f"坐标 ({self.click_coord[1]}, {self.click_coord[0]}) 强度变化")
            self.ax.set_xticks(range(0, len(y_values), 10))
            self.canvas.draw()
            
        except Exception as e:
            print(f"绘图错误: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("图像分析工具")
    app = ImageAnalysisApp(root)
    app.grid(row=0, column=0, sticky='nsew')
    root.mainloop()