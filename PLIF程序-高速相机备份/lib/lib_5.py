# calculation.py
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

def calculate_matrices(params):
    """
    独立计算函数
    输入参数结构：
    {
        "images": f, o, fo图像
        "f_range": (min, max),
        "o_range": (min, max),
        "ellipse": (cx, cy, major, minor),
        "ellipse_whole": (major, minor),
        "processes": int
        "global_mix_ratio": float
    }
    返回四个矩阵：(abs_matrix, me1_matrix, me2_matrix, me3_matrix)
    """
    images = params['images']
        
    # 生成椭圆掩模
    cx, cy, major, minor = params['ellipse']
    mask = np.zeros(images[0].shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (major, minor), 0, 0, 360, (255,), -1)
    ys, xs = np.where(mask > 0)
    coords_me = (xs, ys)

    major_whole, minor_whole = params['ellipse_whole']
    mask = np.zeros(images[0].shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (major_whole, minor_whole), 0, 0, 360, (255,), -1)
    ys, xs = np.where(mask > 0)
    coords_abs = (xs, ys)

    '''
    # 显示掩模（添加文字提示）
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像以便添加文字
    text = f"Center: ({cx}, {cy}), Axes: ({major}, {minor})"
    cv2.putText(mask_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 显示窗口（可调整大小）
    cv2.namedWindow("Ellipse Mask")
    cv2.imshow("Ellipse Mask", mask_display)
    cv2.waitKey(0)  # 按任意键关闭窗口
    cv2.destroyAllWindows()
    '''
    
    # 准备任务参数
    f_shifts = range(*params['f_range'])
    o_shifts = range(*params['o_range'])

    # 初始化进度条（总任务量为F平移量数量）
    pbar = tqdm(total=len(f_shifts), desc="计算进度", unit="F平移量")

    # 多进程计算
    with Pool(processes=params.get('processes', cpu_count())) as pool:
        # 提交所有任务
        futures = []
        for f in f_shifts:
            future = pool.apply_async(
                process_f_shift,
                args=(f, o_shifts, images, coords_abs, coords_me, params["global_mix_ratio"]),
            )
            futures.append(future)
        
        # 按顺序收集结果并更新进度
        results = []
        for future in futures:
            results.append(future.get())
            pbar.update(1)  # 每完成一个F平移量更新进度
        pbar.close()

    '''    
    # 多进程计算
    with Pool(processes=params.get('processes')) as pool:
        tasks = [(f, o_shifts, images, mask) for f in f_shifts]
        results = pool.starmap(process_f_shift, tasks)
    '''
    
    # 重组结果矩阵
    abs_matrix = np.zeros((len(f_shifts), len(o_shifts)))
    me1_matrix = np.zeros_like(abs_matrix)
    me2_matrix = np.zeros_like(abs_matrix)
    me3_matrix = np.zeros_like(abs_matrix)
    
    for i, o_list in enumerate(results):
        for j, (abs_val, me1, me2, me3) in enumerate(o_list):
            abs_matrix[i, j] = abs_val
            me1_matrix[i, j] = me1
            me2_matrix[i, j] = me2
            me3_matrix[i, j] = me3
    me_ptp_matrix = np.ptp(np.stack([me1_matrix, me2_matrix, me3_matrix], axis=2), axis=2)
    return abs_matrix, me_ptp_matrix, me1_matrix, me2_matrix, me3_matrix

def process_f_shift(f_shift, o_shifts, images, coords_abs, coords_me, global_mix_ratio):
    """优化后的处理函数（仅计算mask区域）"""
    f_img, o_img, fo_img = images
    results = []
    for o_shift in o_shifts:
        results.append(calc_result(f_img, o_img, fo_img, f_shift, o_shift, coords_abs, coords_me, global_mix_ratio))
    
    return results

def calc_result(f_img, o_img, fo_img, f_shift, o_shift, coords_abs, coords_me, global_mix_ratio):
    # 计算平移后掩膜并提取有效数据
    xs, ys = coords_me
    f_values = f_img[ys, xs + f_shift].astype(np.float64)
    o_values = o_img[ys, xs + o_shift].astype(np.float64)
    fo_values = fo_img[ys, xs].astype(np.float64)
    # 计算混合效率
    me1 = calc_mixing_efficiency(o_values, fo_values, global_mix_ratio)
    me2 = calc_mixing_efficiency(f_values, fo_values, global_mix_ratio)
    me3 = calc_mixing_efficiency(o_values, f_values + o_values, global_mix_ratio)
    # print(f'O-FO混合效率: {me1}; F-FO混合效率: {me2}; O-F混合效率: {me3}')
    xs, ys = coords_abs
    f_values = f_img[ys, xs + f_shift].astype(np.float64)
    o_values = o_img[ys, xs + o_shift].astype(np.float64)
    fo_values = fo_img[ys, xs].astype(np.float64)
    # 计算差绝对值和
    diff_sum_abs = np.sum(np.abs(f_values + o_values - fo_values))
    # diff_sum_abs = np.sum(np.std(f_values + o_values - fo_values))
    # print(f'f_shift: {f_shift}; o_shift: {o_shift}; 差绝对值和{diff_sum_abs}')
    return diff_sum_abs, me1, me2, me3

def calc_mixing_efficiency(o_values, fo_values, global_mix_ratio):
    valid = fo_values > 0
    o_values = o_values[valid]
    fo_values = fo_values[valid]
    mass_ratio = fo_values / np.sum(fo_values)
    mixing_diff = global_mix_ratio - o_values / fo_values
    result_term = np.where(mixing_diff > 0, mixing_diff / global_mix_ratio, mixing_diff / (global_mix_ratio-1))
    mixing_efficiency = 1-np.sum(result_term * mass_ratio)
    return mixing_efficiency

def shift_image(img, A):# 向左平移为正
    new_img = np.zeros_like(img)
    if A>0:
        new_img[:, :-A] = img[:, A:]
    elif A<0:
        new_img[:, -A:] = img[:, :A]
    else:
        new_img[:, :] = img[:, :]
    return new_img

def exp_plus(path, delta):
    pre, suf = path.split('实验编号')
    exp_no = int(suf.split('.')[0]) + delta
    return f'{pre}实验编号{exp_no}{os.path.splitext(path)[1]}'

def generate_mask_coords(cx, cy, major, minor, shape):
    """生成椭圆掩模的坐标"""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (major, minor), 0, 0, 360, (255,), -1)
    ys, xs = np.where(mask > 0)
    return xs, ys