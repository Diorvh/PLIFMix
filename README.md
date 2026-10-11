# PLIFMix

互击式喷嘴 PLIF 图像处理桌面程序：平均图、像增强器修正、Davis 整理、混合特性及质量分布拟合。

## 启动

使用带 Tcl/Tk 的64位 Python（入口要求3.11或更高；此前在3.14环境验证）。
在 PLIFMix 目录运行：

```powershell
python -m pip install -r requirements.txt
python main.py
```

也可双击安装依赖与启动批处理。Davis 仍需外部人工操作。



2026-10-11 起整理为 PLIFMix；独立公开仓库的重建尚未完成。
请将源码目录命名为 `PLIFMix`；包名与目录名需要一致。

复制整个文件夹可迁移源码；保留 Git 历史需同时复制隐藏的 .git。
新机器需安装依赖；使用自有数据时重新选择外部数据和 Excel 路径。样例路径随软件目录解析，不依赖原电脑盘符。
参数和界面状态统一保存在 runtime/settings.json；内置参数是示例初值，不代表所有实验的默认最优值。

## 样例数据 / Example Data

`样例数据` 文件夹提供三张最终处理的 NPY 图像：F67、FO57、O45，约26 MiB。
对应 DR=1.3、U=25，全局 O/F=1.2786069651741294；样例参数保存在同目录的 `parameters.json`。
正常启动即可使用，无单独演示入口。手动调参和质量分布拟合页面未设置图像目录时默认采用这组数据，已有其他路径不会被覆盖。
随附设置首先打开第4页；切换第5页后，选“拟合”即可使用现有拟合功能。右上角可切换 English / 中文。
其他页面完整保留，但本样例不包含它们所需的原始拍摄图像或标定输入。

The bundled example contains three final processed intensity arrays: F67, FO57 and O45.
Run `python main.py`, then use **4 Mixing** and **5 Distribution**. Select **Fit** on the distribution page to run fitting.
Use the **English** button in the upper-right corner to switch the interface language.
Unconfigured image folders fall back to this example without replacing user-selected folders.
Raw acquisition frames and upstream calibration/Davis inputs are not included; this example demonstrates only the downstream mixing and distribution-fitting workflows.


## 发布状态

当前是公开发布前的源码基线，不是已经完成独立复现验收的正式发行版。
本轮补充中英文界面及一组最终图像样例，没有重跑测试、生成参考输出或上传 GitHub。
依赖版本尚未锁定。本轮不扩展上游原始数据包或其他发布功能。
