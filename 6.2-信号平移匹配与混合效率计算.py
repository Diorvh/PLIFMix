import tkinter as tk
from tkinter import ttk, filedialog
import cv2, json, os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Ensure Axes3D is imported for 3D plotting
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
        # 文件选择部分
        self.setup_ui()
        self.ConfigManager.load_config()

    def setup_ui(self):
        self.base_folder = tk.StringVar()
        self.ConfigManager.register_var('base_folder', self.base_folder)

        frm0 = ttk.Frame(self.root)
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

        # 第二行：F信号平移范围、O信号平移范围
        frm2 = ttk.Frame(self.root)
        frm2.pack(anchor='w')
        ttk.Label(frm2, text="F信号平移范围:").pack(side='left')
        self.f_min = ttk.Entry(frm2, width=7)
        self.ConfigManager.register_entry('f_min', self.f_min)
        self.f_min.pack(side='left')
        ttk.Label(frm2, text="~").pack(side='left')
        self.f_max = ttk.Entry(frm2, width=7)
        self.ConfigManager.register_entry('f_max', self.f_max)
        self.f_max.pack(side='left')
        ttk.Label(frm2, text="O信号平移范围:").pack(side='left')
        self.o_min = ttk.Entry(frm2, width=7)
        self.ConfigManager.register_entry('o_min', self.o_min)
        self.o_min.pack(side='left')
        ttk.Label(frm2, text="~").pack(side='left')
        self.o_max = ttk.Entry(frm2, width=7)
        self.ConfigManager.register_entry('o_max', self.o_max)
        self.o_max.pack(side='left')

        # 第三行：椭圆形计算范围
        frm3 = ttk.Frame(self.root)
        frm3.pack(anchor='w')

        ttk.Label(frm3, text="混合效率椭圆形计算范围圆心:").pack(side='left')
        self.ellipse_x = ttk.Entry(frm3, width=7)
        self.ConfigManager.register_entry('ellipse_x', self.ellipse_x)
        self.ellipse_x.pack(side='left')

        ttk.Label(frm3, text=",").pack(side='left')
        self.ellipse_y = ttk.Entry(frm3, width=7)
        self.ConfigManager.register_entry('ellipse_y', self.ellipse_y)
        self.ellipse_y.pack(side='left')

        ttk.Label(frm3, text="长轴:").pack(side='left')
        self.major_axis = ttk.Entry(frm3, width=7)
        self.ConfigManager.register_entry('major_axis', self.major_axis)
        self.major_axis.pack(side='left')

        ttk.Label(frm3, text="短轴:").pack(side='left')
        self.minor_axis = ttk.Entry(frm3, width=7)
        self.ConfigManager.register_entry('minor_axis', self.minor_axis)
        self.minor_axis.pack(side='left')

        # 第四行：进程数、开始计算按钮
        frm4 = ttk.Frame(self.root)
        frm4.pack(anchor='w')
        
        ttk.Label(frm4, text='全局混合比：').pack(side=tk.LEFT)
        self.ent_globalof = ttk.Entry(frm4, width=5)
        self.ent_globalof.pack(side=tk.LEFT)
        self.ConfigManager.register_entry('globalof', self.ent_globalof)

        ttk.Label(frm4, text="进程数:").pack(side='left')
        self.process_num = ttk.Entry(frm4, width=5)
        self.ConfigManager.register_entry('process_num', self.process_num)
        self.process_num.pack(side='left')
        ttk.Button(frm4, text="开始计算", command=self.start_process).pack(side='left')
        
        # 第五行：结果
        self.res1_lbl = ttk.Label(self.root, text="计算结果：")
        self.res1_lbl.pack(anchor=tk.W)
        self.res2_lbl = ttk.Label(self.root, text="计算结果：")
        self.res2_lbl.pack(anchor=tk.W)

        # 第七行：左侧3D图、右侧3D图
        frm7 = ttk.Frame(self.root)
        frm7.pack(anchor='w')

        self.fig1 = Figure(figsize=(6, 5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111, projection='3d')  # Ensure this is a 3D subplot
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
        self.ax3.set_title("混合效率")
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=frm7)
        self.canvas3.get_tk_widget().grid(row=0, column=2)
        frm7_3 = ttk.Frame(frm7)
        frm7_3.grid(row=1, column=2, sticky='w')
        NavigationToolbar2Tk(self.canvas3, frm7_3)
    
    def select_file(self, path):
        path.set(filedialog.askopenfilename())

    def select_folder(self, path):
        path.set(filedialog.askdirectory())

    def start_process(self):
        # 数据提取
        fo_path = self.base_folder.get()+f'\\实验编号{self.ent_fo_num.get()}.npy'
        f_path = self.base_folder.get()+f'\\实验编号{self.ent_f_num.get()}.npy'
        o_path = self.base_folder.get()+f'\\实验编号{self.ent_o_num.get()}.npy'
        # images = cvimread_list(exp_plus(fo_path, 8), exp_plus(fo_path, 16), fo_path)
        images = cvimread_list(f_path, o_path, fo_path)
        f_range = (int(self.f_min.get()), int(self.f_max.get())+1)
        o_range = (int(self.o_min.get()), int(self.o_max.get())+1)
        ellipse = (int(self.ellipse_x.get()), int(self.ellipse_y.get()), int(self.major_axis.get()), int(self.minor_axis.get()))
        processes = int(self.process_num.get())
        global_mix_ratio = float(self.ent_globalof.get()) / (1 + float(self.ent_globalof.get()))
        params={'images': images, 'f_range': f_range, 'o_range': o_range, 'ellipse': ellipse, 'processes': processes, 'global_mix_ratio': global_mix_ratio}
        results = calculate_matrices(params)
        """更新3D图表"""
        abs_matrix, me_ptp_matrix, me1_matrix, me2_matrix, me3_matrix = results
        min_val, (row, col) = abs_matrix.min(), np.unravel_index(abs_matrix.argmin(), abs_matrix.shape)
        f_range = list(range(f_range[0], f_range[1]))
        o_range = list(range(o_range[0], o_range[1]))
        self.res1_lbl.config(text=f'差绝对值和最小值位置：{(f_range[row], o_range[col])}; 差绝对值和：{abs_matrix[row, col]:.4f}; 混合效率：{me3_matrix[row, col]:.4f}, {me2_matrix[row, col]:.4f}, {me1_matrix[row, col]:.4f}; 混合效率偏差：{me_ptp_matrix[row, col]:.4f}')
        min_val, (row, col) = me_ptp_matrix.min(), np.unravel_index(me_ptp_matrix.argmin(), me_ptp_matrix.shape)
        self.res2_lbl.config(text=f'混合效率偏差最小值位置：{(f_range[row], o_range[col])}; 差绝对值和：{abs_matrix[row, col]:.4f}; 混合效率：{me3_matrix[row, col]:.4f}, {me2_matrix[row, col]:.4f}, {me1_matrix[row, col]:.4f}; 混合效率偏差：{me_ptp_matrix[row, col]:.4f}')

        X, Y = np.meshgrid(range(int(self.f_min.get()), int(self.f_max.get())+1), range(int(self.o_min.get()), int(self.o_max.get())+1))
        
        # 更新左侧图表
        self.ax1.clear()
        self.ax1.plot_surface(X, Y, abs_matrix.T, cmap='viridis') # type: ignore
        self.ax1.set_xlabel('F平移量', fontsize=10, labelpad=10)  # x轴标签
        self.ax1.set_ylabel('O平移量', fontsize=10, labelpad=10)  # y轴标签
        self.ax1.set_zlabel('差绝对值和', fontsize=10, labelpad=10)  # z轴标签 # type: ignore
        self.canvas1.draw()
        
        # 更新右侧图表
        self.ax2.clear()
        self.ax2.plot_surface(X, Y, me_ptp_matrix.T, cmap='plasma') # type: ignore
        self.ax2.set_xlabel('F平移量', fontsize=10, labelpad=10)  # x轴标签
        self.ax2.set_ylabel('O平移量', fontsize=10, labelpad=10)  # y轴标签
        self.ax2.set_zlabel('混合效率极差', fontsize=10, labelpad=10)  # z轴标签 # type: ignore
        self.canvas2.draw()

        # 更新右侧图表
        self.ax3.clear()
        self.ax3.plot_surface(X, Y, ((me1_matrix+me2_matrix+me3_matrix)/3).T, cmap='plasma') # type: ignore
        self.ax3.set_xlabel('F平移量', fontsize=10, labelpad=10)  # x轴标签
        self.ax3.set_ylabel('O平移量', fontsize=10, labelpad=10)  # y轴标签
        self.ax3.set_zlabel('混合效率', fontsize=10, labelpad=10)  # z轴标签 # type: ignore
        self.canvas3.draw()

def on_close():
    app.ConfigManager.save_config()
    root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, '6.2')
    app.pack()
    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()