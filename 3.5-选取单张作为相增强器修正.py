import numpy as np
import os

condition_name = 'Gain65%'
test_name = '800ns.npy'
folder_path = r'E:\Sci\2022.1_impinging_jet\Research\2-PLIF\20250408孔径比影响-2\图像数据\3.1-像增强器修正矩阵'
img_path = os.path.join(folder_path, condition_name, test_name)
img = np.load(img_path)
print(img.max())
img = img / img.max() # 归一化
img = 1 / img # 取倒数
print(img.max(), img.min())
save_name = f'{condition_name}_correct_matrix'
save_path = os.path.join(folder_path, save_name)
np.save(save_path, img)