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
from lib.lib_ConfigManager import ConfigManager

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class App(ttk.Frame):
    def __init__(self, root, name):
        super().__init__(master=root)        
        self.root = root
        self.ConfigManager = ConfigManager(self, name)

        # 初始化图像路径变量
        self.fo_path = tk.StringVar()
        self.f_path = tk.StringVar()
        self.o_path = tk.StringVar()
        self.ConfigManager.register_var('fo_path', self.fo_path)
        self.ConfigManager.register_var('f_path', self.f_path)
        self.ConfigManager.register_var('o_path', self.o_path)

        # 文件选择部分
        self.setup_ui()
        self.ConfigManager.load_config()

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
        self.ConfigManager.register_entry('f_min', self.f_min)
        self.f_min.pack(side='left')
        ttk.Label(frm4, text="~").pack(side='left')
        self.f_max = ttk.Entry(frm4, width=7)
        self.ConfigManager.register_entry('f_max', self.f_max)
        self.f_max.pack(side='left')
        ttk.Label(frm4, text="O信号平移范围:").pack(side='left')
        self.o_min = ttk.Entry(frm4, width=7)
        self.ConfigManager.register_entry('o_min', self.o_min)
        self.o_min.pack(side='left')
        ttk.Label(frm4, text="~").pack(side='left')
        self.o_max = ttk.Entry(frm4, width=7)
        self.ConfigManager.register_entry('o_max', self.o_max)
        self.o_max.pack(side='left')

        # 第五行：椭圆形计算范围
        frm5 = ttk.Frame(self.root)
        frm5.pack(anchor='w')
        ttk.Label(frm5, text="椭圆形计算范围圆心:").pack(side='left')
        self.ellipse_x = ttk.Entry(frm5, width=7)
        self.ConfigManager.register_entry('ellipse_x', self.ellipse_x)
        self.ellipse_x.pack(side='left')
        ttk.Label(frm5, text=",").pack(side='left')
        self.ellipse_y = ttk.Entry(frm5, width=7)
        self.ConfigManager.register_entry('ellipse_y', self.ellipse_y)
        self.ellipse_y.pack(side='left')
        ttk.Label(frm5, text="长轴:").pack(side='left')
        self.major_axis = ttk.Entry(frm5, width=7)
        self.ConfigManager.register_entry('major_axis', self.major_axis)
        self.major_axis.pack(side='left')
        ttk.Label(frm5, text="短轴:").pack(side='left')
        self.minor_axis = ttk.Entry(frm5, width=7)
        self.ConfigManager.register_entry('minor_axis', self.minor_axis)
        self.minor_axis.pack(side='left')

        # 第六行：进程数、开始计算按钮
        frm6 = ttk.Frame(self.root)
        frm6.pack(anchor='w')
        ttk.Label(frm6, text="进程数:").pack(side='left')
        self.process_num = ttk.Entry(frm6, width=5)
        self.ConfigManager.register_entry('process_num', self.process_num)
        self.process_num.pack(side='left')
        ttk.Button(frm6, text="开始计算", command=self.start_process).pack(side='left')

        # 第七行：左侧3D图、右侧3D图
        frm7 = ttk.Frame(self.root)
        frm7.pack(anchor='w')

        self.fig1 = Figure(figsize=(6, 5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111, projection='3d')
        self.ax1.set_title("F信号+O信号与FO信号的差绝对值和")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=frm7)
        self.canvas1.get_tk_widget().grid(row=0, column=0)
        frm7_1 = ttk.Frame(frm7)
        frm7_1.grid(row=1, column=0, sticky='w')
        NavigationToolbar2Tk(self.canvas1, frm7_1)

        self.fig2 = Figure(figsize=(6, 5), dpi=100)
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        self.ax2.set_title("混合效率")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=frm7)
        self.canvas2.get_tk_widget().grid(row=0, column=1)
        frm7_2 = ttk.Frame(frm7)
        frm7_2.grid(row=1, column=1, sticky='w')
        NavigationToolbar2Tk(self.canvas2, frm7_2)

        self.fig3 = Figure(figsize=(6, 5), dpi=100)
        self.ax3 = self.fig3.add_subplot(111, projection='3d')
        self.ax3.set_title("混合效率2")
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=frm7)
        self.canvas3.get_tk_widget().grid(row=2, column=0)
        frm7_3 = ttk.Frame(frm7)
        frm7_3.grid(row=3, column=0, sticky='w')
        NavigationToolbar2Tk(self.canvas3, frm7_3)

        self.fig4 = Figure(figsize=(6, 5), dpi=100)
        self.ax4 = self.fig4.add_subplot(111, projection='3d')
        self.ax4.set_title("混合效率3")
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=frm7)
        self.canvas4.get_tk_widget().grid(row=2, column=1)
        frm7_4 = ttk.Frame(frm7)
        frm7_4.grid(row=3, column=1, sticky='w')
        NavigationToolbar2Tk(self.canvas4, frm7_4)

    
    def select_file(self, path):
        path.set(filedialog.askopenfilename())

    def start_process(self):
        # 数据提取
        images = cvimread_list(self.f_path.get(), self.o_path.get(),self.fo_path.get())
        f_range = (int(self.f_min.get()), int(self.f_max.get())+1)
        o_range = (int(self.o_min.get()), int(self.o_max.get())+1)
        ellipse = (int(self.ellipse_x.get()), int(self.ellipse_y.get()), int(self.major_axis.get()), int(self.minor_axis.get()))
        processes = int(self.process_num.get())
        params={'images': images, 'f_range': f_range, 'o_range': o_range, 'ellipse': ellipse, 'processes': processes, 'global_mix_ratio': 0.5}
        results = calculate_matrices(params)
        self.update_plots(results)

    def update_plots(self, results):
        """更新3D图表"""
        abs_matrix, me1_matrix, me2_matrix, me3_matrix = results
        min_val, (row, col) = abs_matrix.min(), np.unravel_index(abs_matrix.argmin(), abs_matrix.shape)
        print(f'最小值位置：{(list(range(int(self.f_min.get()), int(self.f_max.get())+1))[row], list(range(int(self.o_min.get()), int(self.o_max.get())+1))[col])})')
        print(f'差绝对值和：{abs_matrix[row, col]}; 混合效率：{me1_matrix[row, col], me2_matrix[row, col], me3_matrix[row, col]}')

        X, Y = np.meshgrid(range(int(self.f_min.get()), int(self.f_max.get())+1), range(int(self.o_min.get()), int(self.o_max.get())+1))
        
        # 更新左侧图表
        self.ax1.clear()
        self.ax1.plot_surface(X, Y, abs_matrix.T, cmap='viridis')
        self.ax1.set_xlabel('F平移量', fontsize=10, labelpad=10)  # x轴标签
        self.ax1.set_ylabel('O平移量', fontsize=10, labelpad=10)  # y轴标签
        self.ax1.set_zlabel('差绝对值和', fontsize=10, labelpad=10)  # z轴标签
        self.canvas1.draw()
        
        # 更新右侧图表
        self.ax2.clear()
        self.ax2.plot_surface(X, Y, me1_matrix.T, cmap='plasma')
        self.ax2.set_xlabel('F平移量', fontsize=10, labelpad=10)  # x轴标签
        self.ax2.set_ylabel('O平移量', fontsize=10, labelpad=10)  # y轴标签
        self.ax2.set_zlabel('混合效率1', fontsize=10, labelpad=10)  # z轴标签
        self.canvas2.draw()

        self.ax3.clear()
        self.ax3.plot_surface(X, Y, me2_matrix.T, cmap='plasma')
        self.ax3.set_xlabel('F平移量', fontsize=10, labelpad=10)
        self.ax3.set_ylabel('O平移量', fontsize=10, labelpad=10)
        self.ax3.set_zlabel('混合效率2', fontsize=10, labelpad=10)
        self.canvas3.draw()

        self.ax4.clear()
        self.ax4.plot_surface(X, Y, me3_matrix.T, cmap='plasma')
        self.ax4.set_xlabel('F平移量', fontsize=10, labelpad=10)
        self.ax4.set_ylabel('O平移量', fontsize=10, labelpad=10)
        self.ax4.set_zlabel('混合效率3', fontsize=10, labelpad=10)
        self.canvas4.draw()

def on_close():
    app.ConfigManager.save_config()
    root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, '6.2')
    app.pack()
    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()