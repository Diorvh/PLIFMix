from lib.lib_cv import *
from time import time
from multiprocessing import Pool, cpu_count
import os

def average_img(file_chunk): # 取绿色通道
    """单进程求一堆图像的平均图，如果三通道就转换为单通道"""
    accum = None
    count = 0
    for path in file_chunk:
        img = cvimread(path)
        '''
        工业相机专用去底噪
        '''
        # print(np.sum(img==2))
        img[img<=2] = 0
        # print(np.sum(img==2))
        # os.system('pause')

        if img.dtype == np.uint8:
            img = img.astype(np.float64) * 256
        else:
            img = img.astype(np.float64)
        # 转换三通道到单通道
        if img.ndim == 3:
            # img = np.dot(img, [0.114, 0.587, 0.299])
            img = img[:,:,1] # 取绿色通道

        accum = img if accum is None else accum + img
        count += 1
    return accum, count

def average_img_parallel(file_paths, num_processes = cpu_count()):
    """多进程求一堆图像的平均图，如果三通道就转换为单通道"""
    num_processes = min(num_processes, len(file_paths))
    chunk_size = max(1, len(file_paths) // num_processes + 1)
    # chunk_size = 50
    file_chunks = [file_paths[i:i+chunk_size] 
                  for i in range(0, len(file_paths), chunk_size)]
    # 创建进程池（自动管理资源）
    start_time = time()
    with Pool(processes=num_processes) as pool:
        results = pool.map(average_img, file_chunks)
    print(f'平均图计算耗时：{time() - start_time:.3f}s')
    # 收集结果
    total_accum = 0
    total_count = 0
    for accum, count in results:
        if accum is not None:
            total_accum = accum if total_accum is None else total_accum + accum
            total_count += count
    return total_accum / total_count

def calculate_folder_average_img(folders: Union[str, List[str]], save_name = None, num_processes = cpu_count()):
    '''
    输入文件夹路径，或者一个list的文件夹路径，计算其中所有图像的平均图，然后输出到save_name中（float64格式）
    '''
    # 判断输入类型
    if isinstance(folders, str):
        folders = [folders]
    if save_name == None:
        save_name = os.path.basename(folders[0])
    try:
        # 读取所有文件
        all_files = []
        for folder in folders:
            for f in os.listdir(folder):
                all_files.append(os.path.join(folder, f))
        print(f'图片个数：{len(all_files)}')
        avg_img = average_img_parallel(all_files, num_processes)
        # 计算最终平均并保存
        imsave_float(avg_img, os.path.dirname(folders[0]), save_name)
    except Exception as e:
        print(e)