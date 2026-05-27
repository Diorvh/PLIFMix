import tkinter as tk
from tkinter import ttk, filedialog
import cv2,os,re
import numpy as np
from PIL import Image, ImageTk
from lib.lib_5 import shift_image
from lib.lib_cv import *
from lib.lib_histogram_frame import *

QUAD = [{'x': 100, 'y': 200}, {'x': 100, 'y': 500}, {'x': 700, 'y': 500}, {'x': 700, 'y': 200}]
COLORMAP = BlackedJet_colormap()#BlueWhiteRed_colormap()

class TabEditor(ttk.Notebook):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.bind("<Double-1>", self.edit_tab_name)
        self.bind("<<NotebookTabChanged>>", self.root.load_image)
        
    def edit_tab_name(self, event):
        index = self.index("@%d,%d" % (event.x, event.y))
        new_name = tk.simpledialog.askstring("修改名称", "输入新名称:", parent=self) # type: ignore
        if new_name: self.tab(index, text=new_name)

    def next_tab(self, event):
        tabs = self.tabs()  # 获取所有TAB对象[5](@ref)
        current = self.index(self.select())  # 当前选中索引[5](@ref)
        new_idx = (current + 1) % len(tabs)  # 循环切换
        self.select(tabs[new_idx])  # 切换到新TAB[5](@ref)

    def prev_tab(self, event):
        tabs = self.tabs()
        current = self.index(self.select())
        new_idx = (current - 1) % len(tabs)
        self.select(tabs[new_idx])

    def select_idx(self, idx):
        return self.nametowidget(self.tabs()[idx])
    
    def get_current_tab_widget(self):
        return self.nametowidget(self.select())

class FileSelectFrame(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root=root
        ttk.Button(self, text="选择图片", command=self.file_select).pack(side=tk.LEFT)
        self.lbl_path = ttk.Label(self, text="未选择图片")
        self.lbl_path.pack(side=tk.LEFT)
        self.filename = None

    def file_select(self):
        self.filename = filedialog.askopenfilename()
        self.lbl_path.config(text=self.filename)
        self.root.root.load_image()
    
    def update_filename(self, filename):
        self.filename = filename
        self.lbl_path.config(text=self.filename)

    def get_filename(self):
        return self.filename

class ImageProcessorApp(ttk.Frame):
    def __init__(self, root):
        super().__init__(master=root)
        self.root = root
        self.pack()
        self.initialize_variables()
        self.setup_ui()

    def initialize_variables(self):
        self.image_path = None
        self.original = None
        self.display = None
        self.is_pseudo = False
        self.cmap_max = 255
        self.gain = 1
        self.denoise = 0
        self.rotate_angle = 0
        self.shift = 0
        self.zoom_out = 1
        self.center = (100, 100)
        self.selecting_center = False
        self.crosshair = False
        self.ellipse = (100, 100)
        self.show_ellipse = False
        self.quad = None
        self.show_quad = False
        self.save_cbar = tk.BooleanVar()

    def setup_ui(self):
        # 第一行：文件选择
        frm1 = ttk.Frame(self)
        frm1.pack(fill=tk.X)
        self.btn_import = ttk.Button(frm1, text="一键导入图片", command=self.import_exp_all)
        self.btn_import.pack(side=tk.LEFT)
        self.btn_last_image = ttk.Button(frm1, text='上一张', command=lambda: self.move_image(-1))
        self.btn_last_image.pack(side=tk.LEFT)
        self.btn_next_image = ttk.Button(frm1, text='下一张', command=lambda: self.move_image(1))
        self.btn_next_image.pack(side=tk.LEFT)

        self.notebook = TabEditor(self)
        self.notebook.pack(fill=tk.X)
        for _ in range(10):
            tab = FileSelectFrame(self.notebook)
            self.notebook.add(tab, text=f"实验编号{_}")
        self.root.bind("<Left>", self.notebook.prev_tab)  # 绑定左键[2](@ref)
        self.root.bind("<Right>", self.notebook.next_tab)  # 绑定右键[2](@ref)

        # 第二行：伪彩图控制
        frm2 = ttk.Frame(self)
        frm2.pack(fill=tk.X)
        ttk.Label(frm2, text="增益：").pack(side=tk.LEFT)
        self.ent_gain = ttk.Entry(frm2, width=5)
        self.ent_gain.insert(0, "1")
        self.ent_gain.pack(side=tk.LEFT)
        self.ent_gain.bind('<Return>', self.update_gain)
        self.btn_pseudo = ttk.Button(frm2, text="显示伪彩图", command=self.toggle_pseudo)
        self.btn_pseudo.pack(side=tk.LEFT)
        ttk.Label(frm2, text="伪彩数值上限：").pack(side=tk.LEFT)
        self.ent_max = ttk.Entry(frm2, width=5)
        self.ent_max.insert(0, "255")
        self.ent_max.pack(side=tk.LEFT)
        self.ent_max.bind('<Return>', self.update_cmap_max)
        ttk.Checkbutton(frm2, text="保存ColorBar", variable=self.save_cbar).pack(side=tk.LEFT)

        # 第三行：平移控制
        frm3 = ttk.Frame(self)
        ttk.Label(frm3, text="左右平移量：").pack(side=tk.LEFT)
        self.ent_shift = ttk.Entry(frm3, width=5)
        self.ent_shift.pack(side=tk.LEFT)
        self.ent_shift.insert(0,"0")
        self.ent_shift.bind('<Return>', self.shift_image)
        ttk.Label(frm3, text="缩小倍数：").pack(side=tk.LEFT)
        self.ent_zoom = ttk.Entry(frm3, width=5)
        self.ent_zoom.pack(side=tk.LEFT)
        self.ent_zoom.insert(0,"1")
        self.ent_zoom.bind('<Return>', self.toggle_zoom)
        ttk.Label(frm3, text="去噪阈值：").pack(side=tk.LEFT)
        self.ent_denoise = ttk.Entry(frm3, width=5)
        self.ent_denoise.pack(side=tk.LEFT)
        self.ent_denoise.insert(0,"0")
        self.ent_denoise.bind('<Return>', self.update_denoise)
        ttk.Label(frm3, text="旋转角度：").pack(side=tk.LEFT)
        self.ent_rotate = ttk.Entry(frm3, width=5)
        self.ent_rotate.pack(side=tk.LEFT)
        self.ent_rotate.insert(0,"0")
        self.ent_rotate.bind('<Return>', self.update_rotate)
        frm3.pack(fill=tk.X)

        # 第四行：中心线与椭圆控制
        frm4 = ttk.Frame(self)
        frm4.pack(fill=tk.X)
        ttk.Label(frm4, text='线宽：').pack(side=tk.LEFT)
        self.ent_line_width = ttk.Entry(frm4, width=5)
        self.ent_line_width.insert(0, "1")
        self.ent_line_width.pack(side=tk.LEFT)
        ttk.Label(frm4, text="中心点坐标：").pack(side=tk.LEFT)
        self.ent_cx = ttk.Entry(frm4, width=5)
        self.ent_cx.insert(0, "0")
        self.ent_cx.pack(side=tk.LEFT)
        self.ent_cx.bind('<Return>', self.update_center)
        self.ent_cy = ttk.Entry(frm4, width=5)
        self.ent_cy.insert(0, "0")
        self.ent_cy.pack(side=tk.LEFT)
        self.ent_cy.bind('<Return>', self.update_center)
        self.btn_cross = ttk.Button(frm4, text="显示十字线", command=self.toggle_crosshair)
        self.btn_cross.pack(side=tk.LEFT)

        ttk.Label(frm4, text="椭圆长短轴：").pack(side=tk.LEFT)
        self.ent_major = ttk.Entry(frm4, width=5)
        self.ent_major.insert(0, "100")
        self.ent_major.pack(side=tk.LEFT)
        self.ent_major.bind('<Return>', self.update_ellipse)
        self.ent_minor = ttk.Entry(frm4, width=5)
        self.ent_minor.insert(0, "100")
        self.ent_minor.pack(side=tk.LEFT)
        self.ent_minor.bind('<Return>', self.update_ellipse)
        self.btn_ellipse = ttk.Button(frm4, text="显示椭圆", command=self.toggle_ellipse)
        self.btn_ellipse.pack(side=tk.LEFT)

        # 第五行：四边形区域划定
        frm5 = ttk.Frame(self)
        frm5.pack(fill=tk.X)
        ttk.Label(frm5, text='四边形区域：').pack(side=tk.LEFT)
        for i in range(1,5):
            ttk.Label(frm5, text=f'点{i}:').pack(side=tk.LEFT)
            for j in ['x', 'y']:
                setattr(self, f'ent_p{i}{j}', ttk.Entry(frm5, width=5))
                getattr(self, f'ent_p{i}{j}').pack(side=tk.LEFT)
                getattr(self, f'ent_p{i}{j}').insert(0, str(QUAD[i-1][j]))
                getattr(self, f'ent_p{i}{j}').bind('<Return>', self.update_quad)
        self.btn_quad = ttk.Button(frm5, text='显示四边形', command=self.toggle_quad)
        self.btn_quad.pack(side=tk.LEFT)

        # 第六行：状态信息
        frm6 = ttk.Frame(self)
        self.lbl_info = ttk.Label(frm6, text="坐标: (0, 0) 值: 0")
        self.lbl_info.pack(side=tk.LEFT)
        ttk.Button(frm6, text="保存图片", command=self.save_image).pack(side=tk.LEFT)
        frm6.pack(fill=tk.X)

        # 第七行：图像显示
        frm7 = ttk.Frame(self)
        frm7.pack(fill=tk.X)
        self.canvas = tk.Canvas(frm7)
        self.canvas.pack(side=tk.LEFT, expand=False, fill='none')
        self.canvas.bind("<Motion>", self.update_mouse_info)
        self.histogram = ImageHistogramFrame(frm7)
        self.histogram.pack(side=tk.LEFT)

    def move_image(self, delta):
        path = self.notebook.get_current_tab_widget().get_filename() # 获取当前tab的filename
        dir_path, filename = os.path.dirname(path), os.path.basename(path)
        name, ext = os.path.splitext(filename)
        new_name = f'{int(name)+delta}{ext}'
        self.notebook.get_current_tab_widget().update_filename(os.path.join(dir_path, new_name))
        self.load_image()
        
    def import_exp_all(self):
        folder = filedialog.askdirectory()
        files = os.listdir(folder)
        files = [f for f in files if f.endswith('.tif') and '实验编号' in f] # 筛选出tif文件
        for f in files:
            match = re.search(r'实验编号(\d+)', f)
            if match:
                idx = int(match.group(1))
            tab = self.notebook.select_idx(idx)
            tab.filename = os.path.join(folder, f)
            tab.lbl_path.config(text=tab.filename)

    def load_image(self, event = None):# 读取图像，统一为16bit单通道图像int/float
        path = self.notebook.get_current_tab_widget().get_filename()# 获取当前tab的filename
        if path == None:
            return
        elif path.endswith('.npy'):
            img = np.load(path) # 暂存为float并且不还原
        else :
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
            if img.dtype==np.uint8:
                img=img.astype(np.uint16)*256
        if len(img.shape) == 3: 
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = img[:,:,1] # 直接取中间通道当作单通道图像
        self.original = img # 读入后处理为16单通道作为原始图像
        self.image_path = path
        self.update_display()

    def rotate_img(self, img, angle):
        # 获取图像尺寸
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # 顺时针旋转：OpenCV 中旋转矩阵默认逆时针为正，因此传入 -angle
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # 计算旋转后新画布大小（避免裁剪）
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # 调整旋转矩阵的平移部分，使图像居中
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]

        # 执行仿射变换
        img = cv2.warpAffine(img, rot_mat, (new_w, new_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))  # 对于带透明通道的图像用全0填充
        return img
    def update_display(self):
        if self.original is None: return
        img = self.original.copy()
        # 不是8位就转换8位
        if img.dtype != np.uint8:
            img = img / 256
        img[img<self.denoise] = 0 # 简单的去噪处理
        img = self.rotate_img(img, -self.rotate_angle)
        # 处理增益
        if self.gain == 0:
            self.gain = 256 / img.max()
        img = np.clip(img*self.gain, 0, 255).astype(np.uint8)
        # 处理伪彩图或直接转三通道
        if self.is_pseudo:
            img = np.clip(img, 0, self.cmap_max) / self.cmap_max * 255
            img = cv2.applyColorMap(img.astype(np.uint8), COLORMAP)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # 处理平移
        img = shift_image(img, int(self.ent_shift.get()))
        # 绘制图形
        if self.crosshair:
            cv2.line(img, (self.center[0], 0), (self.center[0], img.shape[0]), (0,0,255), int(self.ent_line_width.get()))
            cv2.line(img, (0, self.center[1]), (img.shape[1], self.center[1]), (0,0,255), int(self.ent_line_width.get()))
        if self.show_ellipse:
            cv2.ellipse(img, self.center, (int(self.ellipse[0]), int(self.ellipse[1])),0, 0, 360, (0,0,255), int(self.ent_line_width.get()))
        if self.show_quad:
            cv2.polylines(img, [self.quad], isClosed=True, color=(0, 0, 255), thickness=int(self.ent_line_width.get()))
        self.display = img
        

        # 处理缩放
        h, w = img.shape[:2]
        if self.zoom_out != 1:
            w=int(w/self.zoom_out)
            h=int(h/self.zoom_out)
            img = cv2.resize(img, (w, h))
        self.canvas.config(width=w,height=h)
        
        # 显示图像
        self.show_image(img)
        self.histogram.update_image(self.original.copy(), self.quad)

    def show_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.delete("all")
        self.canvas.create_image(int(self.canvas['width'])//2,int(self.canvas['height'])//2,image=self.tk_img)
        self.notebook.focus_force()

    # 伪彩图切换功能
    def toggle_pseudo(self):
        if self.original is None: return
        self.is_pseudo = not self.is_pseudo
        self.btn_pseudo.config(text="关闭伪彩图" if self.is_pseudo else "显示伪彩图")
        self.update_display()

    # 图像平移功能
    def shift_image(self, event):
        self.shift = int(self.ent_shift.get())
        self.update_display()

    # 缩放控制功能
    def toggle_zoom(self, event):
        self.zoom_out = float(self.ent_zoom.get())
        self.update_display()

    # 中心点选择功能
    def update_center(self, event):
        self.center=(int(self.ent_cx.get()),int(self.ent_cy.get()))
        self.update_display()

    # 十字线控制
    def toggle_crosshair(self):
        self.crosshair = not self.crosshair
        self.btn_cross.config(text="关闭十字线" if self.crosshair else "显示十字线")
        self.update_display()

    # 椭圆控制
    def toggle_ellipse(self):
        self.show_ellipse = not self.show_ellipse
        self.btn_ellipse.config(text="关闭椭圆" if self.show_ellipse else "显示椭圆")
        self.update_display()

    # 椭圆控制
    def update_ellipse(self, event):    
        major = int(self.ent_major.get())
        minor = int(self.ent_minor.get())
        self.ellipse = (major, minor)
        self.update_display()

    # 四边形控制
    def toggle_quad(self):
        self.show_quad = not self.show_quad
        self.btn_quad.config(text="关闭四边形" if self.show_quad else "显示四边形")
        self.update_display()

    # 四边形控制
    def update_quad(self, event):
        points_list = []
        for i in range(1, 5):
            x = int(getattr(self, f'ent_p{i}x').get())
            y = int(getattr(self, f'ent_p{i}y').get())
            points_list.append([x, y])
        self.quad = np.array(points_list)
        self.update_display()

    # 颜色映射上限更新
    def update_cmap_max(self, event):
        self.cmap_max = int(self.ent_max.get())
        self.update_display()

    # 颜色映射上限更新
    def update_gain(self, event):
        self.gain = float(self.ent_gain.get())
        self.update_display()

    def update_denoise(self, event):
        self.denoise = float(self.ent_denoise.get())
        self.update_display()

    def update_rotate(self, event):
        self.rotate_angle = float(self.ent_rotate.get())
        self.update_display()

    # 实时鼠标信息更新
    def update_mouse_info(self, event):
        if self.original is None: return
        # 计算实际图像坐标
        x = event.x * self.zoom_out
        y = event.y * self.zoom_out
        # 计算原始数据坐标（考虑平移）
        orig_x = int(x) + self.shift
        orig_y = int(y)
        value = 0
        if (0 <= orig_x < self.original.shape[1]) and (0 <= orig_y < self.original.shape[0]):
            value = self.original[orig_y, orig_x]
        
        self.lbl_info.config(text=f"坐标: ({int(x)}, {int(y)}) 值: {value};{value//256}")

    def create_colorbar(self, height, width, colormap, scale_min, scale_max, padding_top, padding_bottom):
        # 绘制刻度（考虑10个刻度标记）
        height=height-padding_top-padding_bottom
        num_steps = 5
        step_size = height / num_steps
        font_scale = 1  # 字体大小
        font_thickness = 3  # 字体粗细

        # 创建一个纯灰度渐变图像
        gradient = np.linspace(255, 0, height, dtype=np.uint8)
        gradient = np.tile(gradient, (width, 1)).T  # 转置以使其垂直

        # 应用颜色映射
        colorbar = cv2.applyColorMap(gradient, colormap)

        # 在彩色条上方和下方添加白色留白
        total_height = height + padding_top + padding_bottom
        padded_colorbar = np.full((total_height, width, 3), 255, dtype=np.uint8)
        padded_colorbar[padding_top:padding_top + height, :] = colorbar

        # 添加刻度和标签
        scale_image = np.full((total_height, width + 104, 3), 255, dtype=np.uint8)  # +30像素用于刻度
        scale_image[:, :width] = padded_colorbar  # 将彩色条添加到刻度图像中

        for i in range(num_steps + 1):
            y = int(padding_top + height - i * step_size)
            cv2.line(scale_image, (width, y), (width + 5, y), (0, 0, 0), 2)

            # 添加标签
            label = scale_min + (scale_max - scale_min) * (i / num_steps)
            cv2.putText(scale_image, f'{label:.0f}', (width + 10, y + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        return scale_image

    # 图像保存功能
    def save_image(self):
        # 生成保存路径
        assert self.image_path is not None
        assert self.display is not None
        dir_name = os.path.dirname(self.image_path)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        if self.gain != 1:
            base_name += f"-增益{self.gain}"
        if self.is_pseudo:
            base_name += "-伪彩图"
            if self.cmap_max != 255:
                base_name += f"-伪彩上限{self.cmap_max}"
        if self.shift != 0:
            base_name += f"-向左平移{self.shift}"
        if self.crosshair:
            base_name += f"-十字线{self.center}"
        if self.show_ellipse:
            base_name += f"-椭圆{self.center}-{self.ellipse}"
        save_path = os.path.join(dir_name, f"{base_name}-绘制后.tif")

        # 处理colorbar
        if self.save_cbar.get():
            cbar_img= self.create_colorbar(self.display.shape[0], 60, COLORMAP, 0, self.cmap_max, 50, 50)
            # 拼接图像
            self.display = np.hstack([self.display, cbar_img])

        # 使用OpenCV保存支持中文路径
        ext = '.tif'
        ret, buf = cv2.imencode(ext, self.display)
        if ret:
            with open(save_path, 'wb') as f:
                f.write(buf.tobytes())

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()