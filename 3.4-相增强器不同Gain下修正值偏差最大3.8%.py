import numpy as np
import os


def create_circular_mask(shape, radius):
    """生成圆形布尔掩码，True表示在圆内"""
    rows, cols = shape
    # 计算几何中心（浮点数坐标）
    center_y = 600
    center_x = 640
    print(center_y,center_x)
    
    # 生成网格坐标
    y, x = np.ogrid[:rows, :cols]
    
    # 计算每个位置到中心的欧氏距离
    distances = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    
    # 生成掩码：距离 <= 半径
    return distances <= radius

def compute_masked_minmax(matrices, radius=550):
    """计算每个位置的最大/最小值，仅处理圆形区域内"""
    # 校验输入
    if len(matrices) == 0:
        raise ValueError("矩阵列表不能为空")
    shape = matrices[0].shape
    if not all(m.shape == shape for m in matrices):
        raise ValueError("所有矩阵必须形状相同")
    
    # 生成掩码
    mask = create_circular_mask(shape, radius)
    
    # 堆叠矩阵并计算全局最大/最小值
    stacked = np.stack(matrices)  # 形状: (4, H, W)
    max_vals = np.max(stacked, axis=0)
    min_vals = np.min(stacked, axis=0)
    ptp=max_vals-min_vals
    # 应用掩码，圆外设为NaN
    max_vals[~mask] = 0
    max_index = np.argmax(max_vals)
    max_index = np.unravel_index(max_index, max_vals.shape)
    print(max_index)
    max_vals[~mask] = 10000000
    min_vals[~mask] = 10000000
    ptp[~mask]=0
    ptp_index = np.argmax(ptp)
    ptp_index = np.unravel_index(ptp_index, ptp.shape)
    print(ptp_index)
    
    return max_vals, min_vals,ptp

folder=r'E:\实验数据\荧光测试\20250301\相增强器修正'
filename=['Gain60%.npy','Gain65%.npy','Gain70%.npy','Gain75%.npy']
k=[]
for i in filename:
    path=os.path.join(folder,i)
    k.append(np.load(path))
    print(np.load(path)[272,207])

vmax,vmin,ptp=compute_masked_minmax(k,600)
print(np.max(ptp/vmax))

#计算结果，大约差3.8%