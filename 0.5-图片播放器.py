import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import glob
import numpy as np

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("图片查看器")
        
        # 变量初始化
        self.folder_path = tk.StringVar()
        self.image_range = tk.StringVar()
        self.current_image = tk.StringVar()
        self.current_idx = 0
        self.total_images = 0
        self.images = []
        self.playing = False
        self.play_direction = 1  # 1前进, -1后退
        self.fps = tk.IntVar(value=24)
        self.gain = tk.IntVar(value=1)
        
        # 第一行：选择文件夹
        tk.Button(root, text="选择文件夹", command=self.select_folder).grid(row=0, column=0, padx=5, pady=5)
        tk.Label(root, textvariable=self.folder_path).grid(row=0, column=1, columnspan=3, sticky='w', padx=5)
        
        # 第二行：显示图片范围和当前图片
        tk.Label(root, textvariable=self.image_range).grid(row=1, column=0, columnspan=2, padx=5)
        tk.Label(root, textvariable=self.current_image).grid(row=1, column=2, columnspan=2, padx=5)
        tk.Label(root, text='增益').grid(row=1, column=4, padx=5)
        tk.Entry(root, textvariable=self.gain, width=10).grid(row=1, column=5, padx=5)
        
        # 第三行：跳转功能
        tk.Button(root, text="跳转到", command=self.jump_to_image).grid(row=2, column=0, padx=5, pady=5)
        self.entry = tk.Entry(root, width=10)
        self.entry.grid(row=2, column=1, padx=5)
        
        # 第四行：帧率设置
        tk.Label(root, text="帧率：").grid(row=3, column=0, padx=5, pady=5)
        tk.Entry(root, textvariable=self.fps, width=10).grid(row=3, column=1, padx=5)
        
        # 第五行：控制按钮
        tk.Button(root, text="后退", command=lambda: self.start_play(-1)).grid(row=4, column=0, padx=5, pady=5)
        tk.Button(root, text="后退一张", command=lambda: self.image_inc(-1)).grid(row=4, column=1, padx=5, pady=5)
        tk.Button(root, text="暂停", command=self.pause_play).grid(row=4, column=2, padx=5, pady=5)
        tk.Button(root, text="前进一张", command=lambda: self.image_inc(1)).grid(row=4, column=3, padx=5, pady=5)
        tk.Button(root, text="前进", command=lambda: self.start_play(1)).grid(row=4, column=4, padx=5, pady=5)
        
        # 第六行：画布
        self.canvas = tk.Canvas(root, width=800, height=600, bg='white')
        self.canvas.grid(row=5, column=0, columnspan=4, padx=5, pady=5)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)
            png_files = sorted(glob.glob(os.path.join(folder, "*.png")))
            self.images = png_files
            self.total_images = len(png_files)
            self.current_idx = 0
            self.image_range.set(f"图片范围：0-{self.total_images-1}")
            self.update_current_image_display()
            self.show_image()

    def show_image(self):
        if self.images and 0 <= self.current_idx < self.total_images:
            img = Image.open(self.images[self.current_idx])
            img = np.array(img) * self.gain.get()
            img = img[:,:,1]
            self.tk_img = ImageTk.PhotoImage(Image.fromarray(np.uint8(img)))
            self.canvas.create_image(img.shape[1] // 2, img.shape[0] // 2, image=self.tk_img)
            self.update_current_image_display()

    def update_current_image_display(self):
        self.current_image.set(f"当前图片：{self.current_idx}")

    def jump_to_image(self):
        try:
            idx = int(self.entry.get())
            if 0 <= idx < self.total_images:
                self.current_idx = idx
                self.show_image()
        except:
            pass

    def image_inc(self, delta):
        if 0 <= self.current_idx + delta < self.total_images:
            self.current_idx += delta
            self.show_image()

    def start_play(self, direction):
        self.playing = True
        self.play_direction = direction
        self.play_images()

    def pause_play(self):
        self.playing = False

    def play_images(self):
        if not self.playing:
            return
            
        # 检查是否到达边界
        if (self.play_direction == 1 and self.current_idx >= self.total_images - 1) or \
           (self.play_direction == -1 and self.current_idx <= 0):
            self.playing = False
            return
            
        # 更新当前索引
        self.current_idx += self.play_direction
        self.show_image()
        
        # 安排下一帧
        if self.playing:
            delay = int(1000 / self.fps.get())
            self.root.after(delay, self.play_images)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()