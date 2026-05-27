# calculation.py
import cv2
from matplotlib.pylab import f
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import openpyxl

PIXEL_SIZE = 0.14 # mm/pixel
DIAMETER_FUEL = 0.5025 # mm

def calculate_matrices(params):
    """
    独立计算函数
    输入参数结构：
    {
        "images": f, o, fo图像
        "f_range": (min, max),
        "o_range": (min, max),
        "ellipse": (cx, cy, major, minor),
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
                args=(f, o_shifts, images, coords_me, params["global_mix_ratio"]),
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

def process_f_shift(f_shift, o_shifts, images, coords_me, global_mix_ratio):
    """优化后的处理函数（仅计算mask区域）"""
    f_img, o_img, fo_img = images
    results = []
    cache_f = shift_image(f_img, f_shift)
    for o_shift in o_shifts:
        cache_o = shift_image(o_img, o_shift)
        results.append(calc_result(cache_f, cache_o, fo_img, coords_me, global_mix_ratio))
    return results

def calc_result(f_img, o_img, fo_img, coords_me, global_mix_ratio):
    # 计算平移后掩膜并提取有效数据
    xs, ys = coords_me
    f_values = f_img[ys, xs].astype(np.float64)
    o_values = o_img[ys, xs].astype(np.float64)
    fo_values = fo_img[ys, xs].astype(np.float64)
    # 计算混合效率
    me1, _ = calc_mixing_efficiency(o_values, fo_values, global_mix_ratio)
    me2, _ = calc_mixing_efficiency(f_values, fo_values, 1-global_mix_ratio)
    me3, _ = calc_mixing_efficiency(o_values, f_values + o_values, global_mix_ratio)
    # 计算差绝对值和
    diff_sum_abs = np.sum(np.abs(f_values + o_values - fo_values))
    return diff_sum_abs, me1, me2, me3

def calc_mixing_efficiency(o_values, fo_values, global_mix_ratio): #o/(f+o)
    valid = fo_values > 0
    mass_ratio = np.zeros_like(fo_values)
    mixing_diff = np.zeros_like(fo_values)
    
    mass_ratio[valid] = fo_values[valid] / np.sum(fo_values)
    mixing_diff[valid] = global_mix_ratio - o_values[valid] / fo_values[valid]
    result_term = np.where(mixing_diff > 0, mixing_diff / global_mix_ratio, mixing_diff / (global_mix_ratio-1)) * mass_ratio
    mixing_efficiency = 1-np.sum(result_term)
    return mixing_efficiency, result_term

def calc_flowrate_contour(img, thres_contour):
    # xs, ys = coords
    # values = img[ys, xs].astype(np.float64)
    values = img.flatten()
    sorted_values = np.sort(values)[::-1]
    prefix_sum = np.cumsum(sorted_values)
    index = np.searchsorted(prefix_sum, thres_contour * prefix_sum[-1])
    contour_value = sorted_values[index]
    
    h, w = img.shape
    boundary_points = set()
    for i in range (h):
        for j in range(w):
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0<=ni<h and 0<=nj<w:
                    if (img[i,j] - contour_value) * (img[ni,nj] - contour_value) < 0:# 穿过等值线
                        boundary_points.add((i,j))
                        boundary_points.add((ni,nj))
    return contour_value, boundary_points

def fit_center_line(img, Up_limit, Down_limit, max_k, cx, cy, rotate_angle):
    # 记录原始坐标
    py_origin, px_origin = [], []
    for i in range(Up_limit, Down_limit+1):
        top_k_indices = np.argsort(img[i])[-max_k:]
        if 0 not in img[i][top_k_indices]: # 如果>0的有效点超过k个，则平均值有效，否则无效
            py_origin.append(i)
            mean_index = np.mean(top_k_indices)
            px_origin.append(mean_index)
    py_origin = np.array(py_origin)
    px_origin = np.array(px_origin)
    # 坐标变换，旋转点
    point_x = -(py_origin - cy)
    point_y = -(px_origin - cx)
    point_x, point_y = rotate_points_clockwise(point_x, point_y, rotate_angle)
    '''
    wb = openpyxl.Workbook()
    ws = wb.active
    for i in range(len(point_x)):
        ws.cell(row=i+1, column=1, value=point_x[i])
        ws.cell(row=i+1, column=2, value=point_y[i])
    wb.save(r'D:\工作文件同步\20250826孔径比影响\fit_points.xlsx')
    '''
    # 开展拟合
    def fit_function(x, a, b): # y = a*sqrt(x^2+b^2)-a*b
        return a * np.sqrt(x**2 + b**2) - a * b
    a, b, r_squared, y_pred = func_fit_2(point_x, point_y, fit_function)
    _, y_pred_rotated = rotate_points_clockwise(point_x, y_pred, -rotate_angle)
    
    return a, b, r_squared, (py_origin, np.round(px_origin).astype(np.int32)), (py_origin, np.round(-y_pred_rotated + cx).astype(np.int32)), point_x

def func_fit_2(x, y, fit_function):
    from scipy.optimize import curve_fit
    # 使用curve_fit进行拟合
    try:
        popt, pcov = curve_fit(fit_function, x, y, p0=[1, 1])
        # 提取拟合参数
        a_fit, b_fit = popt
    except:
        a_fit, b_fit = 0, 0
    # 计算拟合值
    y_fit = fit_function(x, a_fit, b_fit)
    # 计算R²
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return a_fit, b_fit, r_squared, y_fit

def func_fit_3(x, y, fit_function):
    mask = y > 0
    from scipy.optimize import curve_fit
    # 使用curve_fit进行拟合
    try:
        popt, pcov = curve_fit(fit_function, x[mask], y[mask], p0=[1, 1, 1])
        # 提取拟合参数
        a_fit, b_fit, c_fit = popt
    except:
        a_fit, b_fit, c_fit = 0, 0, 0
    # 计算拟合值
    y_fit = fit_function(x, a_fit, b_fit, c_fit)
    # 计算R²
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return a_fit, b_fit, c_fit, r_squared, y_fit

def find_optimal_rho_robust(objective, rho_bounds, n_grid):
    """
    鲁棒的优化方法，结合网格搜索和局部优化
    """
    
    # 第一步：粗糙的网格搜索
    rho_grid = np.linspace(rho_bounds[0], rho_bounds[1], n_grid)
    grid_errors = [objective(rho) for rho in rho_grid]
    best_grid_idx = np.argmin(grid_errors)
    best_grid_rho = rho_grid[best_grid_idx]
    
    # 第二步：在最佳点附近进行局部优化
    if best_grid_idx == 0:
        local_bounds = [rho_grid[0], rho_grid[1]]
    elif best_grid_idx == len(rho_grid)-1:
        local_bounds = [rho_grid[-2], rho_grid[-1]]
    else:
        local_bounds = [rho_grid[best_grid_idx-1], rho_grid[best_grid_idx+1]]
    from scipy import optimize
    result = optimize.minimize_scalar(
        objective, 
        bounds=local_bounds, 
        method='bounded'
    )
    
    return result.x, result.fun

def plot_black_points(img, points):
    y, x = zip(*points)
    img[y, x] = 0
    return img

def rotate_points_clockwise(x, y, angle_degrees):
    """
    将点(x,y)顺时针旋转指定角度
    """
    theta = np.radians(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    x_rotated = x * cos_theta + y * sin_theta
    y_rotated = -x * sin_theta + y * cos_theta
    
    return x_rotated, y_rotated

'''
def calc_mixing_efficiency_old(o_values, fo_values, global_mix_ratio): #o/(f+o)
    valid = fo_values > 0
    o_values = o_values[valid]
    fo_values = fo_values[valid]
    mass_ratio = fo_values / np.sum(fo_values)
    mixing_diff = global_mix_ratio - o_values / fo_values
    result_term = np.where(mixing_diff > 0, mixing_diff / global_mix_ratio, mixing_diff / (global_mix_ratio-1)) * mass_ratio
    mixing_efficiency = 1-np.sum(result_term)
    return mixing_efficiency, result_term
'''

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