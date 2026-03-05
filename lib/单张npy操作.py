import numpy as np
import os
from lib_cv import *

test_name = '标定板.tif'
folder_path = r'E:\Sci\2022.1_impinging_jet\Research\2-PLIF\20250408孔径比影响-2\图像数据\4-角度修正后'
img_path = os.path.join(folder_path,  test_name)
img = cvimread(img_path)
np.save(img_path, img)