from lib.lib_avg_img import *
import os

if __name__ == '__main__':
    main_folder = r'D:\实验数据\20260209MMHNTO混合\像增强器标定'
    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)
        
        for sub_folder_name in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder_name)
            calculate_folder_average_img(sub_folder_path, f'{sub_folder_name}')