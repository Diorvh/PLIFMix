import cv2
import os
import numpy as np
from tkinter import Tk, filedialog
import multiprocessing as mp
import sys
from functools import partial

def process_chunk(image_paths):
    sum_img = None
    count = 0
    for idx, img_path in enumerate(image_paths, 1):
        try:
            # 处理中文路径的读取
            with open(img_path, 'rb') as f:
                buf = f.read()
            img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            
            # 转换为单通道
            if img.ndim == 3 and img.shape[2] == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            
            # 初始化累加器
            if sum_img is None:
                sum_img = np.zeros_like(img_gray, dtype=np.uint64)
            else:
                if img_gray.shape != sum_img.shape:
                    raise ValueError(f"尺寸不一致: {img_path}")
            
            sum_img += img_gray.astype(np.uint64)
            count += 1
            
            # 每处理100张输出进度
            if idx % 100 == 0:
                print(f"进程 {mp.current_process().name} 已处理 {idx} 张图片")
                
        except Exception as e:
            print(f"处理 {img_path} 时出错: {str(e)}")
    
    return sum_img, count

def main_function(processes=24):
    # 创建Tkinter根窗口并隐藏
    root = Tk()
    root.withdraw()
    
    # 选择文件夹
    folder_path = filedialog.askdirectory(title="请选择包含TIFF图像的文件夹")
    if not folder_path:
        print("未选择文件夹")
        return
    
    # 获取所有TIFF文件
    tiff_files = []
    for root_dir, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith('.tif'):
                tiff_files.append(os.path.join(root_dir, f))
    
    if not tiff_files:
        print("未找到TIFF文件")
        return
    
    # 分割任务到各个进程
    chunks = np.array_split(tiff_files, processes)
    
    # 创建进程池
    with mp.Pool(processes=processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # 合并结果
    total_sum = None
    total_count = 0
    for sum_img, count in results:
        if sum_img is not None:
            total_count += count
            if total_sum is None:
                total_sum = sum_img.astype(np.uint64)
            else:
                total_sum += sum_img
    
    if total_sum is None or total_count == 0:
        print("没有有效图像可处理")
        return
    
    # 计算平均图
    avg_img = np.round(total_sum / total_count).astype(np.uint16)
    
    # 构造输出路径
    folder_name = os.path.basename(folder_path)
    parent_dir = os.path.dirname(folder_path)
    output_path = os.path.join(parent_dir, f"{folder_name}.tif")
    
    # 处理中文路径保存
    _, buf = cv2.imencode('.tif', avg_img)
    with open(output_path, 'wb') as f:
        f.write(buf)
    
    print(f"平均图已保存至: {output_path}")

if __name__ == '__main__':
    mp.freeze_support()
    main_function(processes=20)