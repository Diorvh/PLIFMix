from lib.lib_tkinter import *
from lib.lib_avg_img import *

START = 2
END = 10
EXP_NUM = [21, 23, 24]#[3,25,34,5,26,43,7,27,44,9,28,36,29,38,32,40,33,47]## 

FOLDER_PATH = r'D:\实验数据\20260209MMHNTO混合\原始图像\实验编号'
save_dir = r'D:\工作文件同步\20260117MMHNTO互击混合\图像处理\1-去底噪平均图'

def extract(ls):
    res = ls[200:600]+ls[3200:3700]+ls[3800:4400]+ls[4500:5000]
    return res

def process(EXP_NUM):
    # folder_path = select_folder()
    folder_path = FOLDER_PATH+str(EXP_NUM)
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.PNG')]
    # print(png_files)
    print(f"找到 {len(png_files)} 个png文件")
    if png_files[0] != '0000.PNG':
        print('重命名中')
        for i, file in enumerate(png_files):
            old_path = os.path.join(folder_path, file)
            new_path = os.path.join(folder_path, f"{i:04d}.PNG")
            os.rename(old_path, new_path)

    png_files = [f for f in os.listdir(folder_path) if f.endswith('.PNG')]
    start = int(START*500)
    end = int(END*500)
    png_files = [folder_path+'\\'+f for f in png_files[start:end]]
    # png_files = [folder_path+'\\'+f for f in png_files]
    # png_files = extract(png_files)
    print(png_files[0], png_files[-1])
    avg_img = average_img_parallel(png_files)
    imsave_float(avg_img, save_dir, os.path.basename(folder_path))
if __name__ == '__main__':
    for i in EXP_NUM:
        process(i)