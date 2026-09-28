# PLIFv2

互击式喷嘴 PLIF 图像处理桌面程序：平均图、像增强器修正、Davis 整理、混合特性及质量分布拟合。

## 启动

使用带 Tcl/Tk 的64位 Python（入口要求3.11或更高；此前在3.14环境验证）。
在 PLIFv2 目录运行：

```powershell
python -m pip install -r requirements.txt
python main.py
```

也可双击安装依赖与启动批处理。Davis 仍需外部人工操作。



当前真实目录：`D:\工作文件同步\Codes\PLIFv2`。

复制整个文件夹可迁移源码；保留 Git 历史需同时复制隐藏的 .git。
新机器需安装依赖，并重新选择 settings 内外部数据和 Excel 路径。
参数和界面状态统一保存在 runtime/settings.json；实验原始数据不包含在仓库内。
