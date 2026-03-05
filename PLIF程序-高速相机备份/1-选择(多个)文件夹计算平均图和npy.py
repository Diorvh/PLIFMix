import os
import numpy as np
from tkinter import filedialog
from multiprocessing import Pool, cpu_count
import sys
from time import time
from lib.lib_cv import *
from lib.lib_tkinter import *

def process_files(file_chunk):
    """处理文件块的子进程函数"""
    accum = None
    count = 0
    for path in file_chunk:
        img = cvimread(path).astype(np.float64)
        # 转换三通道到单通道
        if img.ndim == 3:
            img = np.dot(img, [0.299, 0.587, 0.114])
        if accum is None:
            accum = np.zeros_like(img, dtype=np.float64)
        accum += img
        count += 1
    return accum, count

def main():
    # 选择文件夹
    folders, save_name = select_multiple_folders()
    all_files = []
    for folder in folders:
        for f in os.listdir(folder):
            if f.endswith('.tif'):
                all_files.append(os.path.join(folder, f))
    print(len(all_files))
    # 创建进程池
    num_processes = min(NUM_PROCESSES, len(all_files))
    chunk_size = max(1, len(all_files) // num_processes)
    file_chunks = [all_files[i:i+chunk_size] 
                  for i in range(0, len(all_files), chunk_size)]
    print(num_processes)
    # 创建进程池（自动管理资源）
    start_time = time()
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_files, file_chunks)
    print(f'计算耗时：{time() - start_time:.3f}s')
    # 收集结果
    total_accum = None
    total_count = 0
    for accum, count in results:
        if accum is not None:
            total_accum = accum if total_accum is None else total_accum + accum
            total_count += count
    if total_accum is not None:
        avg_img = total_accum / total_count
    # 计算最终平均并保存
    imsave_float(avg_img, os.path.dirname(folders[0]), os.path.basename(save_name))

if __name__ == "__main__":
    NUM_PROCESSES=20
    if sys.platform.startswith('win'):
        from multiprocessing import freeze_support
        freeze_support()
    main()