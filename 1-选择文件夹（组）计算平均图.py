import os
import numpy as np
from tkinter import filedialog
from multiprocessing import Pool, cpu_count
import sys
from time import time
from lib.lib_cv import *
from lib.lib_tkinter import *
from lib.lib_avg_img import *
'''
def calculate_average_img_part():
    EXP_NUM = 36
    folder_path = r'E:\实验数据\荧光测试\20250407孔径比影响-2\原始图像\实验编号'+str(EXP_NUM)
    print(folder_path)
    save_name = f'实验编号{EXP_NUM}'
    file_range = -4000, -1

    all_files = []
    for f in os.listdir(folder_path)[file_range[0]:file_range[1]]:
        all_files.append(os.path.join(folder_path, f))

    # file_range = 0, 1000
    # folder = r'E:\实验数据\荧光测试\20250407孔径比影响-2\原始图像\实验编号10'
    # folders =[]
    # for num in [23,33,12,24,34,25,35,14,26,36]:
    #     folders.append(folder.replace('10', str(num)))
    # print(folders)

    save_name = '总底噪'
    all_files = []
    for folder in folders:
        for f in os.listdir(folder)[file_range[0]: file_range[1]]:
            all_files.append(os.path.join(folder, f))
    print(all_files[0])
    print(all_files[-1])

    print(f'图片个数：{len(all_files)}')
    avg_img = average_img_parallel(all_files, NUM_PROCESSES)
    imsave_float(avg_img * 256, os.path.dirname(folders[0]), os.path.basename(save_name))
'''

if __name__ == "__main__":
    # if sys.platform.startswith('win'):
    #     from multiprocessing import freeze_support
    #     freeze_support()

    folder_path = select_folder()
    folders = os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path,folders[0])):
        for folder in folders:
            calculate_folder_average_img(os.path.join(folder_path, folder))
    else:
        calculate_folder_average_img(folder_path)
