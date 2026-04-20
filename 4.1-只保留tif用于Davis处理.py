import os
if __name__ == "__main__":
    folder_path = r'D:\工作文件同步\20260416MMHNTO模拟液雾化混合-3\图像处理\2-像增强器修正图-4 - 只含增益tif'
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # if file.endswith('.npy') or 'GAIN' in file:
            if 'GAIN' not in file:
                os.remove(file_path)