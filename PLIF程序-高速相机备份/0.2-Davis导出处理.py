import os
import re
import tkinter as tk
from tkinter import filedialog
from lib.lib_cv import *

'''
Davis导出设置：Export tif as word；export to external folder: Root手动选择，Folder为默认，Filename为<attr: LoadFile>
'''
def select_folder():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title="请选择文件夹A")
    root.destroy()
    return folder_path

def process_images():# 把图片从文件夹里提取出来并重命名
    # 选择文件夹A
    folder_a = select_folder()
    # 遍历文件夹A中的所有条目
    for entry in os.listdir(folder_a):
        entry_path = os.path.join(folder_a, entry)#文件夹名称
        if os.path.isdir(entry_path):
            file_name=os.listdir(entry_path)[0]
            # D__工作文件同步_20250301孔径比影响-1_图片数据_4.1-Davis角度修正_实验编号14-GAIN20_ImageCorrection_01_B00001.im70001
            # 仅处理子文件夹
            name = ''
            for p in file_name.split('_'):
                if '实验编号' in p:
                    name = p
                    break
            src_path = os.path.join(entry_path, file_name)
            dest_path = os.path.join(folder_a, name+'.tif') # 实验编号XX-GAINXX.tif
            try:
                # 移动文件并重命名
                os.rename(src_path, dest_path)
                # 删除空子文件夹
                os.rmdir(entry_path)
                print(f"已移动：{src_path} → {dest_path}")
            except Exception as e:
                print(f"操作失败：{str(e)}")

    #还原增益
    for i in os.listdir(folder_a):
        if 'GAIN' in i and '实验编号' in i and os.path.isfile(os.path.join(folder_a, i)):
            img = cvimread(os.path.join(folder_a, i))
            gain = int(os.path.splitext(i)[0].split('GAIN')[1])
            img = img.astype(np.float64) / gain
            np.save(os.path.join(folder_a, f'{i.split('-')[0]}.npy'), img)

if __name__ == "__main__":
    process_images()