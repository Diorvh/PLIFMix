import os, cv2
from lib.lib_cv import *
from lib.lib_tkinter import *
from multiprocessing import Pool

def plot_interactive_line(x, y):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore
    import tkinter as tk
    root = tk.Tk()
    plt.switch_backend('TkAgg')
    fig = plt.Figure(figsize=(8, 5)) # type: ignore
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker='o', linestyle='--', color='#2E86C1', markersize=8, linewidth=2, alpha=0.8)
    ax.set_title("Interactive Line Plot", pad=20)
    ax.set_xlabel("X Axis", labelpad=10)
    ax.set_ylabel("Y Axis", labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    root.mainloop()

def get_img_max(img_path):
    return BGR2GRAY_float(cvimread(img_path)).max()

def parallel(file_paths):
    with Pool(processes = NUM_PROCESSOR) as pool:
        return pool.map(get_img_max, file_paths)
    
def plot_max_value():
    folder = select_folder()
    print(folder)
    file_paths = os.listdir(folder)[file_range[0]:file_range[1]]
    file_paths = [os.path.join(folder, f) for f in file_paths]
    print(file_paths[0])
    print(file_paths[-1])

    max_value = parallel(file_paths)
    img_no = list(range(file_range[0], file_range[1]))
    plot_interactive_line(img_no, max_value)

if __name__ == '__main__':
    file_range = 0, 1000
    NUM_PROCESSOR = 23
    plot_max_value()