import numpy as np

def find_topk_values(arr, k=10):
    """
    查找二维数组中最大的k个值及其坐标
    时间复杂度: O(n + klogk) 适用于大型矩阵
    
    参数:
        arr : np.ndarray - 二维数组
        k   : int       - 需要查找的最大值数量
    
    返回:
        list of tuples - 每个元素为(数值, 行索引, 列索引)，按值降序排列
    """
    # 展平数组并获取索引
    flat_indices = np.argpartition(arr.flatten(), -k)[-k:]  # O(n)操作
    partial_sorted = arr.flatten()[flat_indices]
    
    # 二次排序获取精确顺序
    sorted_indices = np.argsort(-partial_sorted)  # O(k log k)
    topk_indices = flat_indices[sorted_indices]
    
    # 转换为二维坐标
    rows, cols = np.unravel_index(topk_indices, arr.shape)
    
    # 组装结果
    return [(arr[r][c], int(r), int(c)) for r, c in zip(rows, cols)]

file_path = r'E:\实验数据\荧光测试\20250407孔径比影响-2\原始图像\总底噪.npy'
img = np.load(file_path)
for i in find_topk_values(img, 100):
    print(i)