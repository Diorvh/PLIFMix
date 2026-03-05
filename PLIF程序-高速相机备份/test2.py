import tkinter as tk
from tkinter import ttk, filedialog
import cv2, json, os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, Manager
from lib.lib_cv import *
from lib.lib_5 import *

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class App(ttk.Frame):
    def __init__(self, root):
        super().__init__(master=root)        
        self.root = root
        self.pack()

        # 参数保存和加载
        self.save_widgets = []
        self.save_var = []
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        # 初始化图像路径变量
        self.fo_path = tk.StringVar()
        self.f_path = tk.StringVar()
        self.o_path = tk.StringVar()
        self.save_var = ['fo_path', 'f_path', 'o_path']

        # 文件选择部分
        self.setup_ui()
        self.load_config()
        
        self.result_abs = self.result_me1 = None  # 用于存储ME1矩阵

    def setup_ui(self):
        # 第一行：F+O信号
        frm1 = ttk.Frame(self.root)
        frm1.pack(anchor='w')
        ttk.Button(frm1, text="选择F+O信号图片", command=lambda: self.select_file(self.fo_path), width=15).pack(side='left')
        ttk.Label(frm1, textvariable=self.fo_path).pack(side='left')

        # 第二行：F信号
        frm2 = ttk.Frame(self.root)
        frm2.pack(anchor='w')
        ttk.Button(frm2, text="选择F信号图片", command=lambda: self.select_file(self.f_path), width=15).pack(side='left')
        ttk.Label(frm2, textvariable=self.f_path).pack(side='left')

        # 第三行：O信号
        frm3 = ttk.Frame(self.root)
        frm3.pack(anchor='w')
        ttk.Button(frm3, text="选择O信号图片", command=lambda: self.select_file(self.o_path), width=15).pack(side='left')
        ttk.Label(frm3, textvariable=self.o_path).pack(side='left')

        # 第四行：F信号平移范围、O信号平移范围
        frm4 = ttk.Frame(self.root)
        frm4.pack(anchor='w')
        ttk.Label(frm4, text="F信号平移范围:").pack(side='left')
        self.f_min = ttk.Entry(frm4, width=7)
        self.save_widgets.append('f_min')
        self.f_min.pack(side='left')
        ttk.Label(frm4, text="~").pack(side='left')
        self.f_max = ttk.Entry(frm4, width=7)
        self.save_widgets.append('f_max')
        self.f_max.pack(side='left')
        ttk.Label(frm4, text="O信号平移范围:").pack(side='left')
        self.o_min = ttk.Entry(frm4, width=7)
        self.save_widgets.append('o_min')
        self.o_min.pack(side='left')
        ttk.Label(frm4, text="~").pack(side='left')
        self.o_max = ttk.Entry(frm4, width=7)
        self.save_widgets.append('o_max')
        self.o_max.pack(side='left')

        # 第五行：椭圆形计算范围
        frm5 = ttk.Frame(self.root)
        frm5.pack(anchor='w')
        ttk.Label(frm5, text="椭圆形计算范围圆心:").pack(side='left')
        self.ellipse_x = ttk.Entry(frm5, width=7)
        self.save_widgets.append('ellipse_x')
        self.ellipse_x.pack(side='left')
        ttk.Label(frm5, text=",").pack(side='left')
        self.ellipse_y = ttk.Entry(frm5, width=7)
        self.save_widgets.append('ellipse_y')
        self.ellipse_y.pack(side='left')
        ttk.Label(frm5, text="长轴:").pack(side='left')
        self.major_axis = ttk.Entry(frm5, width=7)
        self.save_widgets.append('major_axis')
        self.major_axis.pack(side='left')
        ttk.Label(frm5, text="短轴:").pack(side='left')
        self.minor_axis = ttk.Entry(frm5, width=7)
        self.save_widgets.append('minor_axis')
        self.minor_axis.pack(side='left')

        # 第六行：进程数、开始计算按钮
        frm6 = ttk.Frame(self.root)
        frm6.pack(anchor='w')
        ttk.Label(frm6, text="进程数:").pack(side='left')
        self.process_num = ttk.Entry(frm6, width=5)
        self.save_widgets.append('process_num')
        self.process_num.pack(side='left')
        ttk.Button(frm6, text="开始计算", command=self.start_process).pack(side='left')

        # 第七行：左侧3D图、右侧3D图
        frm7 = ttk.Frame(self.root)
        frm7.pack(anchor='w')
        self.left_fig = Figure(figsize=(6, 5), dpi=100)
        self.left_ax = self.left_fig.add_subplot(111, projection='3d')
        self.left_ax.set_title("F+O信号与F+O信号方差")
        self.left_canvas = FigureCanvasTkAgg(self.left_fig, master=frm7)
        self.left_canvas.get_tk_widget().grid(row=0, column=0)
        self.right_fig = Figure(figsize=(6, 5), dpi=100)
        self.right_ax = self.right_fig.add_subplot(111, projection='3d')
        self.right_ax.set_title("混合效率")
        self.right_canvas = FigureCanvasTkAgg(self.right_fig, master=frm7)
        self.right_canvas.get_tk_widget().grid(row=0, column=1)
        # 添加工具栏
        frm7_left = ttk.Frame(frm7)
        frm7_left.grid(row=1, column=0, sticky='w')
        frm7_right = ttk.Frame(frm7)
        frm7_right.grid(row=1, column=1, sticky='w')
        NavigationToolbar2Tk(self.left_canvas, frm7_left)
        NavigationToolbar2Tk(self.right_canvas, frm7_right)

    def select_file(self, path_var):
        filepath = filedialog.askopenfilename(filetypes=[("TIFF文件", "*.tif")])
        if filepath:
            # 使用OpenCV读取支持中文路径
            try:
                with open(filepath, 'rb') as f:
                    img_data = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    path_var.set(filepath)
                else:
                    raise ValueError("无法解码图像文件")
            except Exception as e:
                print(f"读取文件错误: {str(e)}")

    def start_process(self):
        # 数据提取
        images = {'fo': load_image(self.fo_path.get()), 'f': load_image(self.f_path.get()), 'o': load_image(self.o_path.get())}
        f_range = (int(self.f_min.get()), int(self.f_max.get())+1)
        o_range = (int(self.o_min.get()), int(self.o_max.get())+1)
        ellipse = (int(self.ellipse_x.get()), int(self.ellipse_y.get()), int(self.major_axis.get()), int(self.minor_axis.get()))
        processes = int(self.process_num.get())
        params={'images': images, 'f_range': f_range, 'o_range': o_range, 'ellipse': ellipse, 'processes': processes}
        print(params)
        results = calculate_matrices(params)
        self.update_plots(results)


    def update_plots(self, results):
        """更新3D图表"""
        abs_matrix, me1_matrix, _, _ = results
        X, Y = np.meshgrid(range(int(self.f_min.get()), int(self.f_max.get())+1), range(int(self.o_min.get()), int(self.o_max.get())+1))
        
        # 更新左侧图表
        self.left_ax.clear()
        self.left_ax.plot_surface(X, Y, abs_matrix.T, cmap='viridis')
        self.left_ax.set_xlabel('F平移量', fontsize=10, labelpad=10)  # x轴标签
        self.left_ax.set_ylabel('O平移量', fontsize=10, labelpad=10)  # y轴标签
        self.left_ax.set_zlabel('差绝对值和', fontsize=10, labelpad=10)  # z轴标签
        self.left_canvas.draw()
        
        # 更新右侧图表
        self.right_ax.clear()
        self.right_ax.plot_surface(X, Y, me1_matrix.T, cmap='plasma')
        self.right_ax.set_xlabel('F平移量', fontsize=10, labelpad=10)  # x轴标签
        self.right_ax.set_ylabel('O平移量', fontsize=10, labelpad=10)  # y轴标签
        self.right_ax.set_zlabel('混合效率1', fontsize=10, labelpad=10)  # z轴标签
        self.right_canvas.draw()


    def on_close(self):
        """窗口关闭事件处理"""
        self.save_config()
        self.root.destroy()

    def save_config(self):
        """保存配置"""
        config = {
            'var': {key: getattr(self, key).get() for key in self.save_var},
            'widgets': {key: getattr(self, key).get() for key in self.save_widgets}
        }
        with open('5-config.json','w') as f:
            json.dump(config, f)
    
    def load_config(self):
        """加载配置"""
        if os.path.exists('5-config.json'):
            with open('5-config.json','r') as f:
                config = json.load(f)
                for key, value in config['var'].items():
                    getattr(self, key).set(value)
                for key, value in config['widgets'].items():
                    getattr(self, key).insert(0, str(value))

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()