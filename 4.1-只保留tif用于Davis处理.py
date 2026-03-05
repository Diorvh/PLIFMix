import os
if __name__ == "__main__":
    folder_path = r'D:\工作文件同步\20260117MMHNTO互击混合\图像处理\3.1-修正后平均图-去底噪3 - 只含增益tif'
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # if file.endswith('.npy') or 'GAIN' in file:
            if 'GAIN' not in file:
                os.remove(file_path)