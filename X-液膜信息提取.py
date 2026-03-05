import tkinter as tk
from tkinter import ttk, filedialog
import cv2,os, openpyxl
import numpy as np
from PIL import Image, ImageTk
from lib.lib_ConfigManager import ConfigManager
from lib.lib_cv import *

class MeasureSheet(ttk.Frame):
    def __init__(self, root, name):
        super().__init__(master=root)
        self.root = root
        self.ConfigManager = ConfigManager(self, name)
        self.initialize_variables()
        self.setup_ui()
        self.ConfigManager.load_config()
        try:
            self.update_folder()
        except Exception as e:
            print(f"Error updating folder: {e}")

    def initialize_variables(self):
        self.image_no_max = 0
        self.image_no_min = 0
        self.left =0
        self.right =0
        self.top =0
        self.bottom =0
        self.sheet_width =0 
        self.sheet_length =0
        self.img = None
        self.data = []

    def setup_ui(self):
        #第零行：路径选择
        frm0 = ttk.Frame(self)
        frm0.pack(fill=tk.X)
        self.img_folder = tk.StringVar()
        self.res_folder = tk.StringVar()
        self.ConfigManager.register_var('img_folder', self.img_folder)
        self.ConfigManager.register_var('res_folder', self.res_folder)

        frm = ttk.Frame(frm0)
        frm.pack(fill=tk.X)
        ttk.Button(frm, text='选择图像文件夹', command = lambda: self.select_folder(self.img_folder)).pack(side=tk.LEFT)
        ttk.Label(frm, textvariable = self.img_folder).pack(side=tk.LEFT)

        frm = ttk.Frame(frm0)
        frm.pack(fill=tk.X)
        ttk.Button(frm, text='选择结果文件夹', command = lambda: self.res_folder.set(filedialog.askdirectory())).pack(side=tk.LEFT)
        ttk.Label(frm, textvariable = self.res_folder).pack(side=tk.LEFT)

        #第一行：图片信息
        frm1 = ttk.Frame(self)
        frm1.pack(fill=tk.X)

        self.lbl_imagerange = ttk.Label(frm1, text='图像范围：-')
        self.lbl_imagerange.pack(side=tk.LEFT)
        ttk.Label(frm1, text='当前图像编号：').pack(side=tk.LEFT)
        self.image_no = tk.IntVar()
        self.ent_imageno = ttk.Entry(frm1, width=8, textvariable=self.image_no)
        self.ent_imageno.pack(side=tk.LEFT)
        self.ent_imageno.bind('<Return>', self.update_figure)
        
        ttk.Label(frm1, text='提取线坐标：').pack(side=tk.LEFT)
        self.pt_x = tk.IntVar()
        self.ConfigManager.register_var('pt_x', self.pt_x)
        self.ent_pt_x = ttk.Entry(frm1, width=5, textvariable=self.pt_x)
        self.ent_pt_x.pack(side=tk.LEFT)
        self.pt_y = tk.IntVar()
        self.ConfigManager.register_var('pt_y', self.pt_y)
        self.ent_pt_y = ttk.Entry(frm1, width=5, textvariable=self.pt_y)
        self.ent_pt_y.pack(side=tk.LEFT)

        ttk.Label(frm1, text='边界阈值').pack(side=tk.LEFT)
        self.edge_thres = tk.IntVar()
        self.ConfigManager.register_var('edge_thres', self.edge_thres)
        self.ent_edge_thres = ttk.Entry(frm1, width=5, textvariable=self.edge_thres)
        self.ent_edge_thres.pack(side=tk.LEFT)

        ttk.Button(frm1, text='自动计算', command=self.auto_process_edge).pack(side=tk.LEFT)

        #第二行：参数展示
        frm2 = ttk.Frame(self)
        frm2.pack(fill=tk.X)

        self.lbl_left = ttk.Label(frm2, text=f'左线：{self.left}')
        self.lbl_left.grid(row=0, column=0, sticky=tk.W)
        self.lbl_right = ttk.Label(frm2, text=f'右线：{self.right}')
        self.lbl_right.grid(row=0, column=1, sticky=tk.W)
        self.lbl_sheetwidth = ttk.Label(frm2, text=f'液膜宽度：{self.sheet_width} pix')
        self.lbl_sheetwidth.grid(row=0, column=2, sticky=tk.W)
        self.lbl_top = ttk.Label(frm2, text=f'上线：{self.top}')
        self.lbl_top.grid(row=1, column=0, sticky=tk.W)
        self.lbl_bottom = ttk.Label(frm2, text=f'下线：{self.bottom}')
        self.lbl_bottom.grid(row=1, column=1, sticky=tk.W)
        self.lbl_sheetlength = ttk.Label(frm2, text=f'液膜长度：{self.sheet_length} pix')
        self.lbl_sheetlength.grid(row=1, column=2, sticky=tk.W)
        self.presentline = tk.StringVar()
        self.ConfigManager.register_var('presentline', self.presentline)
        ttk.Label(frm2,text='当前线:').grid(row=2, column=0)
        self.lbl_presentline = ttk.Label(frm2,textvariable=self.presentline)
        self.lbl_presentline.grid(row=2, column=1)
        
        #第三行：图像
        frm3 = ttk.Frame(self)
        frm3.pack(fill=tk.X)

        self.canvas = tk.Canvas(frm3)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<Key>', self.on_key_press)
        ttk.Button(frm3, text='设置焦点', command=lambda: self.canvas.focus_set()).pack()
        ttk.Button(frm3, text='保存数据', command=self.save_data).pack()
        self.canvas.pack()

    def select_folder(self, path):
        path.set(filedialog.askdirectory())
        self.update_folder()

    def update_folder(self): #更新文件夹后，更新图像范围前后端
        # 排序
        filenames = os.listdir(self.img_folder.get())
        filenames = [f for f in filenames if f.endswith('.tif')]
        filenames = [int(x[:-4]) for x in filenames]
        filenames.sort()
        # 提取最大值最小值
        self.image_no_min = filenames[0]
        self.image_no_max = filenames[-1]
        # 更新前后端
        self.lbl_imagerange.config(text=f'图像范围：{self.image_no_min} - {self.image_no_max}')
        self.image_no.set(self.image_no_min)
        self.data = []
        self.sheet_length_data = []
        self.update_figure()

    def update_figure(self, event = None): #读取并刷新图像
        self.img = cvimread(os.path.join(self.img_folder.get(), f'{self.image_no.get()}.tif'))
        h, w = self.img.shape[:2]
        self.canvas.config(width=w,height=h)
        if self.img.dtype == np.uint16:
            self.img = (self.img/256).astype(np.uint8)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        cv2.line(self.img, (self.left,0), (self.left,h),(255,0,0),1)
        cv2.line(self.img, (self.right,0), (self.right,h),(255,0,0),1)
        cv2.line(self.img, (0,self.top), (w,self.top),(255,0,0),1)
        cv2.line(self.img, (0,self.bottom), (w,self.bottom),(255,0,0),1)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(self.img))
        self.canvas.delete("all")
        self.canvas.create_image(w//2,h//2,image=self.tk_img)
    
    def on_key_press(self, event):
        if event.char == '4':
            self.presentline.set('left')
        elif event.char == '6':
            self.presentline.set('right')
        elif event.char == '8':
            self.presentline.set('top')
        elif event.char == '5':
            self.presentline.set('bottom')
        elif event.char == '9':
            self.data.append([self.image_no.get(), self.left, self.right, self.top, self.bottom, self.sheet_width, self.sheet_length])
            if self.image_no.get()+1 <= self.image_no_max:
                self.image_no.set(self.image_no.get()+1)
                self.update_figure()
        elif event.char == '7':
            if len(self.data)>0:
                self.data.pop()
            if self.image_no.get()-1 >= self.image_no_min:
                self.image_no.set(self.image_no.get()-1)
                self.update_figure()
        elif event.keysym == 'Up':
            if self.presentline.get() == 'top':
                self.top = max(0, self.top - 1)
                self.lbl_top.config(text=f'上线：{self.top}')
            elif self.presentline.get() == 'bottom':
                self.bottom = max(0, self.bottom - 1)
                self.lbl_bottom.config(text=f'下线：{self.bottom}')
        elif event.keysym == 'Down':
            if self.presentline.get() == 'top':
                self.top = min(self.top + 1, self.canvas.winfo_height())
                self.lbl_top.config(text=f'上线：{self.top}')
            elif self.presentline.get() == 'bottom':
                self.bottom = min(self.bottom + 1, self.canvas.winfo_height())
                self.lbl_bottom.config(text=f'下线：{self.bottom}')
        elif event.keysym == 'Left':
            if self.presentline.get() == 'left':
                self.left = max(0, self.left - 1)
                self.lbl_left.config(text=f'左线：{self.left}')
            elif self.presentline.get() == 'right':
                self.right = max(0, self.right - 1)
                self.lbl_right.config(text=f'右线：{self.right}')
        elif event.keysym == 'Right':
            if self.presentline.get() == 'left':
                self.left = min(self.left + 1, self.canvas.winfo_width())
                self.lbl_left.config(text=f'左线：{self.left}')
            elif self.presentline.get() == 'right':
                self.right = min(self.right + 1, self.canvas.winfo_width())
                self.lbl_right.config(text=f'右线：{self.right}')
        self.update_parameters()
        self.update_figure()

    def on_canvas_click(self, event):
        if 0 <= event.x and event.x <= self.canvas.winfo_width() and 0 <= event.y and event.y <= self.canvas.winfo_height():
            if self.presentline.get() == 'left':
                self.left = event.x
                self.lbl_left.config(text=f'左线：{self.left}')
            elif self.presentline.get() == 'right':
                self.right = event.x
                self.lbl_right.config(text=f'右线：{self.right}')
            elif self.presentline.get() == 'top':
                self.top = event.y
                self.lbl_top.config(text=f'上线：{self.top}')
            elif self.presentline.get() == 'bottom':
                self.bottom = event.y
                self.lbl_bottom.config(text=f'下线：{self.bottom}')
            self.update_parameters()
            self.update_figure()
    
    def update_parameters(self):
        self.sheet_width = self.right - self.left
        self.lbl_sheetwidth.config(text=f'液膜宽度：{self.sheet_width} pix')
        self.sheet_length = self.bottom - self.top
        self.lbl_sheetlength.config(text=f'液膜长度：{self.sheet_length} pix')
    
    def save_data(self):
        wb_target = openpyxl.Workbook()
        ws_target = wb_target.active
        title = [os.path.basename(self.img_folder.get()), '起始编号', self.image_no_min, '结束编号', self.image_no_max]
        for ni, i in enumerate(title):
            ws_target.cell(1, ni+1, i)
        title = ['图像编号', '左线', '右线', '上线', '下线', '液膜宽度', '液膜长度']
        for ni, i in enumerate(title):
            ws_target.cell(2, ni+1, i)
        for ni, i in enumerate(self.data):
            for nj, j in enumerate(i):
                ws_target.cell(ni+3, nj+1, j)
        wb_target.save(os.path.join(self.res_folder.get(), f'{os.path.basename(self.img_folder.get())}液膜信息.xlsx'))

    def auto_process_edge(self):
        left_edge, right_edge = [], []
        for i in range(self.image_no_min, self.image_no_max+1):
            img = cvimread(os.path.join(self.img_folder.get(), f'{i}.tif'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x, y, thres = self.pt_x.get(), self.pt_y.get(), self.edge_thres.get()
            left_point, right_point = x, x
            row_pixels = img[y, :]
            for i in range(x, -1, -1):  # 从x开始向左搜索
                if row_pixels[i] > thres:
                    left_point = i
                    break
            for i in range(x, img.shape[1]):  # 从x开始向右搜索
                if row_pixels[i] > thres:
                    right_point = i
                    break
            left_edge.append(left_point)
            right_edge.append(right_point)
        left_edge = np.array(left_edge)
        right_edge = np.array(right_edge)
        middle = np.array((right_edge + left_edge) / 2)
        
        wb_target = openpyxl.Workbook()
        ws_target = wb_target.active
        title = [os.path.basename(self.img_folder.get()), '起始编号', self.image_no_min, '结束编号', self.image_no_max, '边界阈值', self.edge_thres.get(), '提取点坐标', f'({self.pt_x.get()},{self.pt_y.get()})']
        for ni, i in enumerate(title):
            ws_target.cell(1, ni+1, i)
        title = ['图像编号', '左线', '右线', '中线']
        for ni, i in enumerate(title):
            ws_target.cell(2, ni+1, i)
        for ni, i in enumerate(range(self.image_no_min, self.image_no_max+1)):
            ws_target.cell(ni+3, 1, i)
            ws_target.cell(ni+3, 2, left_edge[ni])
            ws_target.cell(ni+3, 3, right_edge[ni])
            ws_target.cell(ni+3, 4, middle[ni])
        wb_target.save(os.path.join(self.res_folder.get(), f'{os.path.basename(self.img_folder.get())}射流振荡信息.xlsx'))
def on_close():
    app.ConfigManager.save_config()
    root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MeasureSheet(root, "MeasureSheet")
    app.pack(fill=tk.BOTH, expand=True)
    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()