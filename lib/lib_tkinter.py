import tkinter as tk
from tkinter import ttk, filedialog
import os
def select_multiple_folders():
    root = tk.Tk()
    root.title("多选文件夹")
    selected_folders = []
    save_name = ''
    # 创建带滚动条的列表框
    list_frame = ttk.Frame(root)
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
    listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.EXTENDED,)
    scrollbar.config(command=listbox.yview)
    # 操作按钮组
    def add_folder():
        if folder := filedialog.askdirectory(title="选择文件夹"):
            if folder not in selected_folders:
                selected_folders.append(folder)
                listbox.insert(tk.END, folder)
    def remove_selected():
        for i in reversed(listbox.curselection()):
            listbox.delete(i)
            del selected_folders[i]
    def finish_select():
        nonlocal save_name
        save_name = save_entry.get() or os.path.basename(selected_folders[0])
        root.quit()

    save_frame = ttk.Frame(root)
    ttk.Label(save_frame, text='保存文件名称：').pack(side=tk.LEFT)
    save_entry = ttk.Entry(save_frame)
    save_entry.pack(side=tk.LEFT)

    btn_frame = ttk.Frame(root)
    ttk.Button(btn_frame, text="添加文件夹", command=add_folder).pack(pady=5, fill=tk.X)
    ttk.Button(btn_frame, text="删除选中", command=remove_selected).pack(pady=5, fill=tk.X)
    ttk.Button(btn_frame, text="完成", command=finish_select).pack(pady=5, fill=tk.X)
    # 布局组件
    list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    save_frame.pack(side=tk.LEFT, padx=10)
    btn_frame.pack(side=tk.LEFT, padx=10)
    root.mainloop()
    return selected_folders, save_name