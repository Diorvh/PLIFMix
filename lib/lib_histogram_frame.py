import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class ImageHistogramFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.image = None
        
        # 创建输入控件
        frm1 = ttk.Frame(self)
        frm1.pack()
        self.gain_entry = ttk.Entry(frm1)
        self.gain_entry.insert(0, "1.0")
        self.gain_entry.bind('<Return>', self.update_histogram)
        # 在原frm1框架下添加坐标范围控件
        ttk.Label(frm1, text="Gain:").pack(side=tk.LEFT)
        self.gain_entry.pack(side=tk.LEFT, padx=5)
        
        # 新建坐标范围输入框架
        range_frame = ttk.Frame(frm1)
        range_frame.pack(side=tk.LEFT, padx=10)
        
        # X轴范围
        ttk.Label(range_frame, text="X:").grid(row=0, column=0)
        self.xmin_entry = ttk.Entry(range_frame, width=6)
        self.xmin_entry.insert(0, "0")
        self.xmin_entry.grid(row=0, column=1)
        ttk.Label(range_frame, text="-").grid(row=0, column=2)
        self.xmax_entry = ttk.Entry(range_frame, width=6)
        self.xmax_entry.insert(0, "255")
        self.xmax_entry.grid(row=0, column=3)

        # Y轴范围 
        ttk.Label(range_frame, text="Y:").grid(row=0, column=4, padx=(10,0))
        self.ymin_entry = ttk.Entry(range_frame, width=6)
        self.ymin_entry.insert(0, "auto")
        self.ymin_entry.grid(row=0, column=5)
        ttk.Label(range_frame, text="-").grid(row=0, column=6)
        self.ymax_entry = ttk.Entry(range_frame, width=6)
        self.ymax_entry.insert(0, "auto")
        self.ymax_entry.grid(row=0, column=7)
        
        # 绑定回车事件
        for entry in [self.xmin_entry, self.xmax_entry, self.ymin_entry, self.ymax_entry]:
            entry.bind('<Return>', self.update_histogram)

        #创建信息显示
        frm2 = ttk.Frame(self)
        frm2.pack()
        self.avg_label = ttk.Label(frm2, text='平均值：')
        self.avg_label.pack(side=tk.LEFT)
        self.max_label = ttk.Label(frm2, text='最大值：')
        self.max_label.pack(side=tk.LEFT)
        self.min_label = ttk.Label(frm2, text='最小值：')
        self.min_label.pack(side=tk.LEFT)
        
        # 创建Matplotlib图形
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(self.canvas, self)

        # 初始显示
        self.update_histogram()

    def update_image(self, image):
        self.image = image
        self.avg_label.config(text=f'平均值：{np.average(image)}')
        self.max_label.config(text=f'最大值：{image.max()}')
        self.min_label.config(text=f'最小值：{image.min()}')
        self.update_histogram()

    def process_image(self, img, gain):
        """图像处理流程"""
        try:
            # 转换浮点型并应用增益
            processed = img.astype(np.float64) * gain
            # 处理16位图像
            if img.dtype != np.uint8:
                processed /= 256
            # 限幅并转换到8位
            processed = np.clip(processed, 0, 255).astype(np.uint8)
            # 转换到单通道
            if processed.ndim == 3 and processed.shape[2] == 3:
                processed = np.dot(processed[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            return processed
        except:
            pass

    def update_histogram(self, event = None):
        """更新直方图"""
        try:
            gain = float(self.gain_entry.get())
            processed_img = self.process_image(self.image, gain)
            # 清除旧图形
            self.ax.clear()
            # 获取用户输入范围
            xmin = float(self.xmin_entry.get()) if self.xmin_entry.get() != "auto" else 0
            xmax = float(self.xmax_entry.get()) if self.xmax_entry.get() != "auto" else 255
            
            ymin = float(self.ymin_entry.get()) if self.ymin_entry.get() != "auto" else None
            ymax = float(self.ymax_entry.get()) if self.ymax_entry.get() != "auto" else None
            # 绘制新直方图
            self.ax.hist(processed_img.ravel(), bins=256, range=(xmin, xmax), density=True)
            self.ax.set_title("Grayscale Histogram")
            self.ax.set_xlabel("Pixel Value")
            self.ax.set_ylabel("Frequency")
            self.ax.set_xlim(left=xmin, right=xmax)
            if ymin is not None or ymax is not None:
                self.ax.set_ylim(bottom=ymin, top=ymax)
            self.canvas.draw()
        except:
            pass
if __name__ == "__main__":
    # 生成测试图像（16位三通道）
    height, width = 200, 200
    test_image = np.random.randint(0, 65535, (height, width, 3), dtype=np.uint16)
    
    # 创建主窗口
    root = tk.Tk()
    root.title("Image Histogram Viewer")
    
    # 创建并显示框架
    app = ImageHistogramFrame(root)
    app.pack(fill='both', expand=True, padx=10, pady=10)

    app.update_image(test_image)
    
    root.mainloop()