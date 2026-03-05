import numpy as np
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename(filetypes=[("图像矩阵", "*.npy")])
img = np.load(path)
print(img)
print(img.min(), img.max())