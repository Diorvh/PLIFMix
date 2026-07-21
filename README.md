# PLIFv2

互击式喷嘴 PLIF 图像处理程序第二版。

目标是让 `PLIFv2` 文件夹可以独立复制到其他位置后运行，不依赖旧版脚本目录。

## 启动

建议使用 64 位 Python 3.11-3.13，安装时保留 `Tcl/Tk and IDLE` 组件。首次使用可双击：

```text
安装依赖.bat
```

之后双击 `启动PLIFv2.bat` 即可启动。也可在 VSCode 终端直接运行：

在包含 `PLIFv2` 文件夹的上级目录运行：

```powershell
python -m PLIFv2.main
```

如果在 VSCode 中直接打开 `PLIFv2` 文件夹：

```powershell
python main.py
```

## 已完成流程

1. 平均图按实验编号逐个处理，固定提取绿色通道，去噪清零后再求平均。
2. 支持时间与采样频率选帧，以及复选框控制的非连续帧区间。
3. 像增强器修正矩阵应用。
4. 像增强器标定图线性拟合生成修正矩阵。
5. Davis 导出结果整理和 GAIN 还原。
6. F/O 平移搜索和混合效率计算。
7. 单组 F/O/FO 参数的混合效率计算。
8. 平均图计算前自动检查并必要时重命名为 `0000` 序列。
9. Excel 实验增益映射与对应修正矩阵批处理。
10. 像增强器误差热图、单点拟合、区域误差和矩阵比较。
11. Davis 输入准备与输出整理。
12. 6.1 结果导出和独立 6.3 质量分布拟合。
13. 底噪趋势、图像查看、播放器、NPY、GIF、colorbar 等工具。

`Davis` 仍是外部软件。程序中的第 3 页负责把修正结果准备到 Davis 输入目录，以及把 Davis 角度修正后的输出整理回 `.npy/.tif`。

## 搬迁

复制整个 `PLIFv2` 文件夹即可，程序不依赖旧版代码目录。新电脑仍需要安装 Python、Tcl/Tk 和 `requirements.txt` 中的依赖。所有用户配置保存在 `runtime/settings.json`。

## 验证

```powershell
python -m unittest PLIFv2.tests.test_smoke -v
```

## 迁移原则

- GUI 只负责参数输入和触发动作。
- 业务流程在 `services/`。
- 纯计算在 `algorithms/`。
- 图像读写、配置读写在 `core/`。
- 参数模型集中在 `domain/settings.py`。
- 包内导入统一使用 `PLIFv2.*`，保证目录可搬迁。
