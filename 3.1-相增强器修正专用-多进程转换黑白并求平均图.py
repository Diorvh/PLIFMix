import os
import cv2
import numpy as np
from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm

def process_folder(args, folder_entry):
    # 修正参数解包方式
    root_dir, result_dict = args
    idx, folder_name = folder_entry
    folder_path = os.path.join(root_dir, folder_name)
    
    # 初始化存储数组
    img_stack = []
    first_img = None
    
    # 遍历所有TIFF文件
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.tif'):
            file_path = os.path.join(folder_path, file_name)
            
            # 处理中文路径
            try:
                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint16), cv2.IMREAD_UNCHANGED)
            except Exception as e:
                print(f"读取失败: {file_path} - {str(e)}")
                continue

            # 检查并转换通道
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 保存单通道图像
                cv2.imencode('.tif', img)[1].tofile(file_path)
            
            # 确保统一尺寸
            if first_img is None:
                first_img = img
                h, w = img.shape
            elif img.shape != first_img.shape:
                continue
                
            img_stack.append(img)

    if not img_stack:
        return None

    # 转换为numpy数组
    img_stack = np.array(img_stack, dtype=np.uint32)

    # 计算统计量
    avg_img = np.mean(img_stack, axis=0).astype(np.uint16)
    max_img = np.max(img_stack, axis=0).astype(np.uint16)
    min_img = np.min(img_stack, axis=0).astype(np.uint16)
    diff_img = max_img - min_img

    # 保存结果
    output_path = os.path.join(root_dir, f"{folder_name}.tif")
    cv2.imencode('.tif', avg_img)[1].tofile(output_path)
    
    # 记录统计信息
    stats = (
        np.max(diff_img), 
        np.mean(diff_img), 
        np.max(diff_img) / np.mean(diff_img) if np.mean(diff_img) != 0 else 0
    )
    
    # 保持处理顺序
    result_dict[idx] = (folder_name, stats)
    return idx, avg_img.shape

def main():
    root_dir = r'E:\实验数据\荧光测试\20250301\相增强器修正\Gain75%'
    num_processes = 20  # 可调节的进程数
    
    # 获取有序文件夹列表（保留原始顺序）
    folders = [(i, f) for i, f in enumerate(os.listdir(root_dir)) 
              if os.path.isdir(os.path.join(root_dir, f))]
    
    # 创建共享字典保持顺序
    with Manager() as manager:
        result_dict = manager.dict()
        
        # 创建进程池
        with Pool(processes=num_processes) as pool:
            # 修正参数绑定方式
            process_func = partial(process_folder, (root_dir, result_dict))
            results = list(tqdm(
                pool.imap_unordered(process_func, folders),
                total=len(folders),
                desc="处理文件夹"
            ))

        # 按原始顺序输出结果
        ordered_results = [result_dict[k] for k in sorted(result_dict.keys())]
        for folder_name, stats in ordered_results:
            print(f"{os.path.basename(folder_name)} : {stats[0]:.2f} {stats[1]:.2f} {stats[2]:.2f}")

if __name__ == "__main__":
    main()