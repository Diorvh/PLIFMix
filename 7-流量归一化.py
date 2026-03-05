from lib.lib_cv import *
from lib.lib_tkinter import *
import numpy as np
import openpyxl

if __name__ == '__main__':
    folder = select_folder()
    res_max = np.zeros(250, dtype = np.float64)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name) # .npy图像
        img = cvimread(img_path)
        img_sum = np.sum(img)
        img = img / img_sum
        imsave_float(img, folder, os.path.splitext(img_name)[0])
        # 记录最大值
        exp_num = int(os.path.splitext(img_name)[0][4:])
        res_max[exp_num] = np.max(img)

    xlsx_path = os.path.join(folder, '最大值记录.xlsx')
    wb = openpyxl.Workbook()
    ws = wb.active
    for i, value in enumerate(res_max):
        if value != 0:
            ws.cell(row = i, column = 1, value = value)
    wb.save(xlsx_path)