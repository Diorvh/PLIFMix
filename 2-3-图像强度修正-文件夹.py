import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import openpyxl
from lib.lib_cv import *
from lib.lib_tkinter import select_folder

excel_path = r'D:\工作文件同步\20260205乙醇-四氯化碳\20260204乙醇-四氯化碳线性度测试-结果处理.xlsx'
gain_column = 5
correction_folder = r'D:\工作文件同步\20260205乙醇-四氯化碳\像增强器修正'
file_path = r'D:\工作文件同步\20260205乙醇-四氯化碳\1-平均图'

if __name__ == "__main__":
    # 初始化Tkinter
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb.active
    for filename in os.listdir(file_path):
        if not (filename.startswith("实验编号") and filename.endswith(".npy")):
            continue
        core = int(filename[len("实验编号"):-len(".npy")])
        gain_value = None
        for row in ws.iter_rows(min_row=1, max_col=1, values_only=False):
            cell_a = row[0]          # A列单元格
            if cell_a.value == core:
                gain_value = ws.cell(row=cell_a.row, column=gain_column).value
                break
        # 构造 Gain 文件名，例如 "Gain45.npy"
        gain_filename = f"Gain{gain_value}.npy"
        gain_filepath = os.path.join(correction_folder, gain_filename)
        # 读取 Gain 文件（numpy 格式）
        correction_matrix = np.load(gain_filepath)
        print(filename, gain_filename)
        process_image(os.path.join(file_path, filename), correction_matrix, 'multiply')