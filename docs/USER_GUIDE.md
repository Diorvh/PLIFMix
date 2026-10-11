# PLIFMix User Guide

[Project overview](../README.md)

This guide covers operation rather than code architecture. English controls are paired with their Chinese names where useful. Switching language does not rename files or change settings or calculated results.

## Contents

1. [Install and start](#1-install-and-start)
2. [Prepare inputs](#2-prepare-inputs)
3. [Average experimental frames](#3-average-experimental-frames)
4. [Correct the image intensifier response](#4-correct-the-image-intensifier-response)
5. [Prepare and import DaVis files](#5-prepare-and-import-davis-files)
6. [Analyze mixing](#6-analyze-mixing)
7. [Fit intensity distributions](#7-fit-intensity-distributions)
8. [Use the example](#8-use-the-example)
9. [Other tools](#9-other-tools)
10. [Troubleshooting and records](#10-troubleshooting-and-records)

## 1. Install and Start

Use Windows with 64-bit Python 3.11 or later, including Tcl/Tk. Put the complete project in a writable folder; its name need not be `PLIFMix`. Open a terminal there:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Alternatively, use the bundled installation and startup batch files. With several Python installations, ensure both actions use the same interpreter. For an isolated environment, use its interpreter explicitly:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

The batch files use `py`, or fall back to `python`; they do not automatically select this `.venv`.

The window opens maximized. Switch English/Chinese at the upper right. Both rows of page buttons remain available; the second row contains tools, not subordinate processing steps.

On normal exit, parameters and page state are saved in `runtime/settings.json`. Keep that file to retain your configuration. External paths may need to be selected again after moving computers; bundled sample paths resolve relative to the application.

## 2. Prepare Inputs

For new experiments, average frames, correct the intensifier response, perform geometric correction in DaVis, and import the final images. Both Mixing and Distribution then use these final images. If they are already available, start directly at either analysis page.

| Input | Organization |
| --- | --- |
| Raw experiments | Folders such as `实验编号67`, each containing PNG, TIF/TIFF, or BMP frames directly. The averaging page also accepts a single folder with any name. |
| Raw calibration | A root containing `Gain43`, `Gain50`, etc.; each gain folder contains exposure folders such as `500ns` and `10000ns`, each holding its frames. |
| Calibration averages | Exposure-named arrays inside a gain folder, for example `Gain43/500ns.npy`. |
| Correction matrices | `Gain43.npy`, `Gain50.npy`, etc., matching camera gains. |
| Final images | Same-sized, finite, two-dimensional arrays such as `实验编号67.npy`, `实验编号57.npy`, and `实验编号45.npy`. |

F is the F-only image, O the O-only image, and FO the combined image for the same condition. The application cannot infer their physical roles from brightness. The `实验编号` prefix is a file convention even in English; enter only the number in ID fields. Use `.xlsx` for workbook inputs; convert legacy `.xls` files first.

### Common controls

| Control | Meaning |
| --- | --- |
| `...` beside a path | Select the folder or file for that row. |
| Enter in a field | Apply edited values on interactive pages. Wait for completion before exporting. |
| Gain / Auto | Display brightness. Auto maps the current maximum to the display maximum; it is not intensifier calibration. |
| Downscale / 缩小倍数 | Display-size divisor: 1 = native size, 2 = half-size, 0.5 = double-size. Analysis arrays are not resized. |
| Absolute threshold / 去噪阈值 | Before averaging, values at or below this threshold become zero in each frame. Enter 8-bit values for 8-bit images; higher-bit-depth averaging inputs use their native intensity scale. |
| Relative threshold / 相对去噪阈值 | Fraction of an image's maximum: 0.01 means 1%, not an absolute gray value. This affects mixing/fitting data. |
| X / Y | Zero-based pixel column/row. Check the on-image readout when selecting coordinates. |

Color-image averaging uses the green channel. Eight-bit inputs are represented on a 16-bit intensity scale in processing outputs by multiplying by 256. Do not apply this factor again to an existing processed NPY. Keep numerical arrays separate from rendered exports.

## 3. Average Experimental Frames

Open **1 Average / 1 平均图**.

1. Select **Raw image root / 原始图像根目录** and **Average output / 平均图输出目录**.
2. Set **Optimal workers / 最佳进程数** and **Threshold / 去噪阈值**. Optimal workers is the preferred count for one experiment; folder count and local logical CPUs also determine allocation.
3. Choose an input mode below.
4. Click **计算平均图**. The log lists experiments, frame counts, output paths, and image-computation time. This time excludes preparation and final exports.

**Experiment-ID mode:** check **Experiment ID / 实验编号**, select the common root, and enter IDs such as `2,16,30`. All experiments use the same range. Set Start/s, End/s, and Sampling/Hz; 2 to 10 seconds at 500 Hz selects indices 1000 through 4999.

Alternatively, check **Non-contiguous intervals / 非连续区间** and enter `200:600,3200:3700`. These are zero-based sequence positions, start inclusive and end exclusive. Time settings are ignored in this mode. Avoid overlapping intervals: overlapping positions would be counted repeatedly.

**Single-folder mode:** uncheck Experiment ID and select the image folder itself. All supported images directly in it are averaged. Time/interval controls are disabled and the output uses the folder's name.

**Source-file warning:** sequence checking is automatic. If names are not already `0000`, `0001`, etc. in the detected order, the application renames them in place. Work on copies if original names must be retained.

Output normally includes an NPY, a 16-bit TIFF, and a brightness-scaled `-GAIN...tif`. Use the NPY for subsequent quantitative processing. All-zero results do not produce an additional gain-scaled image.

## 4. Correct the Image Intensifier Response

Open **2 Intensifier / 2 像增强器**, then **Workflow / 处理流程**.

### A. Calibration averages

1. Select **Calibration root / 像增强器标定根目录**, containing all `Gainxx` folders.
2. Select **Calibration output root / 标定平均图根目录**.
3. Set Optimal workers and Threshold, then click **计算标定平均图**.

All frames in every gain/exposure folder are averaged. Output mirrors the gain level, for example `calibration-averages/Gain43/500ns.npy`. There is no time-range selection here. The source-sequence renaming warning from step 3 also applies.

### B. Matrix fitting

1. Select one gain's **Calibration averages / 标定平均图目录**, such as `calibration-averages/Gain43`, not the parent containing several gains.
2. Set **Matrix output / 修正矩阵输出目录** to the common matrix destination.
3. Click **Fit and analyze / 拟合并分析**. The plots show the largest-exposure image, a relative-error map, and a pointwise fit.
4. Click the left or middle plot to inspect that pixel's measured-versus-fitted response on the right.
5. Optionally set **ROI radius / 区域分析半径** and click **计算区域平均误差**. The error color limit affects only the plot; 0 selects automatic scaling.
6. Click **Save matrix / 保存修正矩阵**. Repeat for other gains. The source folder name supplies the matrix name, such as `Gain43.npy`.

Exposure names must contain their intended numeric value. Do not mix exposure units or unrelated files in a gain's average folder.

### C. Experimental correction

Use either tab in **Experimental image correction / 实验图像修正**:

- **Excel batch / 按 Excel 批量修正:** select the average folder, matrix folder, gain workbook, and output folder. Enter ID/Gain column numbers starting at 1 (A = 1). The active sheet is read. Gain values are multiplied by 100 when matching matrix names: 0.43 selects `Gain43.npy`. Click Excel batch. Only `.npy` inputs are processed. Unmatched IDs are logged as skipped; missing matrices and duplicate output names produce errors.
- **Single image / 单张修正:** select one NPY image, one NPY matrix, and an output folder; click **Correct image / 修正所选图像**.

Correction multiplies the image by the matrix; dimensions must match. Keep the corrected NPY and gain-scaled TIFF.

**Advanced diagnostics / 高级诊断** is optional. Average check plots raw-frame pixel fluctuations; matrix comparison inspects variation among gain matrices. Neither is required to apply an existing matrix.

## 5. Prepare and Import DaVis Files

Open **3 Davis I/O / 3 Davis 整理**.

### Prepare input

1. Select **Corrected image folder / 像增强器修正后目录** and a separate **Davis input folder / Davis 输入目录**.
2. Click **Prepare Davis input / 准备 Davis 输入**. TIFFs containing `GAIN` in their names are collected into that folder; duplicate names are rejected.
3. Perform geometric correction in external DaVis. PLIFMix does not run or configure DaVis.

### Import output

1. Export corrected images from DaVis, keeping the experiment ID and `GAIN` value in each filename.
2. Select **Davis export folder / Davis 导出目录** and **Output folder / 整理输出目录**.
3. Click **Import Davis output / 整理 Davis 输出**.

The importer divides each export by its filename's gain and produces a new NPY, TIFF, and gain-scaled TIFF. It does not copy the original unprocessed image. Do not divide by the gain manually beforehand.

Provide one export per experiment. A flat folder is supported; for subfolders, only the first supported file in each immediate subfolder is used. Do not put several different exports in one subfolder. Missing gain values and duplicate experiment IDs are rejected. The final NPY folder feeds both analysis pages below.

## 6. Analyze Mixing

Open **4 Mixing / 4 手动调参**.

### Select and adjust

1. Set **Image folder / 图像文件夹** to the final NPY folder.
2. Enter **F**, **FO**, and **O** IDs. All three images are required.
3. Enter **DR** and **U** for the output name, such as `DR1.3_U25`. These labels do not replace the Global O/F setting.
4. Set **Global O/F / 全局 O/F**, relative threshold, and F/O horizontal shifts. Press Enter to apply edits and update results.

Optionally load a **Preset group / 预设数据组** JSON. Previous group, Next group, or a typed group number fills IDs, DR, U, and global O/F. Presets do not contain images; the IDs must exist in the selected folder. The Distribution page provides the same group selection.

**Excel to preset / Excel 转预设** requires a specific layout in the first sheet: numeric IDs in A, Df/Do in B/C, F/FO/O role in E, F speed in M, and global O/F in U. Every three numeric-ID rows must form a complete group. It is not an arbitrary-workbook importer.

### Inspect results

- Use Pseudocolor, Gain/Auto, and Downscale for viewing.
- F/O/FO have separate crosshair coordinates. **Copy F center / 一键填充** copies the F coordinates to the other fields.
- Drawing an ellipse and restricting analysis to it are separate choices. Check **Ellipse ROI only / 仅椭圆区域计算混合效率** to use that region; otherwise the full image is used.
- The first summary row shows measured O/F and the sum of absolute differences. The second reports total-calibrated efficiencies; the third reports efficiencies without total calibration. Both still use the current threshold and shifts.
- **绘图使用标定总量** chooses calibrated or uncalibrated data for plots and exports, without removing the other numeric summary.

The result area shows F/O/FO, sum/difference maps, three mixing-fraction maps, their ME maps, and `mix_curve`. ME maps show weighted deviation contributions to the efficiency calculation; they are not an independent local-efficiency scale.

### Search shifts

Open **Shift search / 平移搜索**, enter F/O bounds and worker count, and click **Search shifts / 搜索平移量**. Three 3D plots show absolute differences, efficiency range, and mean efficiency. Inspect them and manually enter your chosen shifts in the main controls.

### Save

In **Save selected / 一键保存**, tick the images and choose whether saved overlays include crosshairs/ellipse. Wait for completion before saving.

Exports are 8-bit color PNGs in result-named folders beside the input folder. If the input is `study/final-images`, an F export is `study/f/DR1.3_U25-降噪0.01.png`. Selecting `mix_curve` writes curve data as Excel under `study/mix_curve`, not a screenshot. The calibrated-display choice also controls this curve export. Use distinct names to retain multiple parameter variants.

## 7. Fit Intensity Distributions

Open **5 Distribution / 5 质量分布拟合** and select the final folder and F/FO/O IDs. Each thresholded image is independently normalized. This page does not use global O/F total calibration or calculate mixing efficiency.

### Origins and centerlines

1. Set the relative threshold and F/O shifts.
2. Set **Crosshair X/Y**, **Upper/Lower row / 中心线上界/下界**, and an integer **Angle / 旋转角度** above each image separately.
3. Set **Top K / 最大值个数**, the number of bright points used for centerline extraction, not the player's Top-K frame list.
4. Optionally click **自动原点和偏转角**, then visually inspect and correct the proposed values.
5. Inspect **Centerline / 中心线** or **Fitted centerline / 拟合中心线**, which are mutually exclusive. Upper/lower bounds select the fitting data; they do not crop the displayed image to those rows.

Crosshair, Contour, and centerlines are overlays. Line width controls their appearance. **Contour ratio / 轮廓比例** specifies the enclosed intensity fraction used to find a contour, not a peak-brightness fraction. **Reset / 还原** clears overlays and returns to the unstraightened original view; it does not reset all numeric inputs.

### Fit and compare

Choose **Original / 原图**, **Rotate / 旋转**, **Fit / 拟合**, or **|Fit - original| / 拟合-原图绝对值**. **Straighten / 掰直** is a separate option and is unavailable in the rotation-only preview.

Select Fit to calculate transverse, axial, and both two-dimensional fits. The table shows the pure elliptical Student and compact-support models side by side. The fitted rendering uses the compact-support model when available, otherwise the pure elliptical model. Inspect warnings rather than assuming a partial fit is complete.

With Straighten enabled, compare fits directly with the straightened original. The difference view is computed in that frame and can be mapped back to a curved view. Mapped-back fits follow the symmetric fitted centerline and are not new raw measurements.

Returning to Original retains valid fitted results for reuse. Threshold, origin, alignment, row bounds, or angle changes can require recalculation. Wait for completion before saving.

### Save images and fit data

- **Individual Save / 保存:** select **Image output folder / 图像保存目录**, then Save above F, O, or FO. This saves the rendered PNG with a view/overlay suffix, not a floating-point prediction array.
- **Save selected / 一键保存:** select F/O/FO and/or **Fit data / 拟合数据**. The destination is the configured distribution output folder, or `distribution_results` beside the input folder when unconfigured. Fit data creates `*-fit-results.xlsx` and fit-curve PNGs; no existing measurement workbook is needed.
- **Write fit results / 写入拟合结果:** select an existing `.xlsx` in **Results workbook / 拟合结果写入文件**. Save/close it in Excel first and keep a backup. This updates experiment rows rather than creating a generic table.

For the existing-table writer, each requested ID must occur exactly once in column A. Sheet `测量数据` is used when present, otherwise the active sheet. Row 1, columns T:AR, must have these exact headers:

```text
中心x, 中心y, 旋转角度, 上界, 下界,
中心线a, 中心线b, R2,
横向a, 横向b, 横向c, R2,
纵向a, 纵向b, 纵向c, R2,
sx, sy, m, R2,
sx, sy, m, R, R2
```

Headers are unchanged in English mode. Without this layout, use Save selected with Fit data. Prefer plain `.xlsx` files over macro-enabled or feature-rich originals. Spatial fit coordinates and length parameters are in pixels; display labels do not convert them to millimetres or x/D.

## 8. Use the Example

`样例数据` contains final **F67 / FO57 / O45** images for **DR1.3_U25**, global O/F **1.2786069651741294**. No raw sequences, calibration images, or DaVis inputs are supplied.

Unconfigured Mixing/Distribution pages select the sample automatically. If settings point elsewhere, select `样例数据` and the IDs manually. Changing folders does not discard your other parameters.

1. On Mixing, set DR = 1.3, U = 25, relative threshold = 0.01, F/O shifts = 0, and the global O/F above. Inspect both result rows.
2. On Distribution, use the same IDs, threshold = 0.01, shifts = 0, and Top K = 30. The supplied per-image inputs are:

| Image | X | Y | Upper row | Lower row | Angle (degrees) |
| --- | --- | --- | --- | --- | --- |
| F67 | 664 | 426 | 200 | 700 | 4 |
| O45 | 655 | 426 | 150 | 700 | 0 |
| FO57 | 663 | 419 | 100 | 800 | 5 |

3. Select Fit. Inspect Fit and |Fit - original| with Straighten enabled, then compare the curved views.
4. Save only the needed items. Sample distribution exports use `outputs/sample/distribution`, and individual image saves use `outputs/sample/images`. Mixing follows the sibling-folder convention in step 6, even though its settings also contain an output-folder field.

The complete sample inputs are in `样例数据/parameters.json`. These are example choices, not universal defaults. There is no sample measurement workbook; use Fit data to generate a standalone workbook.

## 9. Other Tools

| Page | Use |
| --- | --- |
| Image viewer / 图像查看器 | Choose an Image, or Folder plus ID for `实验编号N.npy`. Selection/Enter refreshes it; Previous/Next changes the numbered file. Drag the divider beside the histogram and use its Matplotlib toolbar. Save view exports the rendering; Batch TIF export accepts `0800-0850`, inclusive, from the image's directory. |
| Image player / 图片播放器 | Select a folder and cache workers; wait for caching. Set FPS and Play/Reverse/Pause, or enter a frame and press Enter. Slow rendering reduces speed rather than intentionally skipping frames. Click a pixel or enter X/Y to update its intensity-versus-frame plot and Top-K list; it does not redraw every playback frame. |
| Noise analysis / 底噪分析 | Select raw frames and an interval; calculate the frame-maximum trend. Images are not modified. |
| NPY tools / NPY矩阵 | Inspect statistics/Top K and save multiplication, addition, maximum normalization, or reciprocal results to another NPY. |
| GIF export / GIF导出 | Choose the folder/output GIF, first/last frame, gain, interval in milliseconds, and maximum width. Numeric names 800 to 850 select `0800` through `0850`, inclusive. Source RGB appearance is retained; grayscale sources stay grayscale. Viewer pseudocolor settings are not inherited. |
| Colorbar | Set dimensions and limits, preview, and save a labeled colorbar. |
| Distributions / 分布对比 | Plot standard symmetric distributions; this does not fit experimental data. |

The player's **Threshold (8-bit)** is an absolute threshold even when values display on the 16-bit scale. Non-contiguous intervals use `start:end,start:end`, end exclusive, and only affect the right trace: excluded positions become zero, without reloading the cache. They do not limit caching or playback. The plot can use Y = 0-65535 or automatic limits.

Viewer/player rotations preserve the full extent for quarter turns and currently accept only multiples of 90 degrees. This differs from integer-angle correction on Distribution.

## 10. Troubleshooting and Records

| Symptom | Action |
| --- | --- |
| Missing Python module | Install requirements with the interpreter used for startup; keep the whole project together. |
| Tcl/Tk window error | Repair/install Python with Tcl/Tk and use a desktop session. |
| No averaging frames | Check the ID checkbox, direct folder contents, sampling frequency, and end-exclusive bounds. Single-folder mode uses all frames. |
| No GIF frames | Choose the frame directory, not its parent. Numeric filename ranges include the end, unlike averaging intervals. |
| Blank/dark/saturated display | Check Gain/Auto, threshold, and pixel readout before concluding the NPY values are incorrect. |
| Missing F/O/FO image | Check exact filenames, all IDs, folder, and equal dimensions. |
| No intensity after thresholding | Lower the relative threshold. A blank image cannot yield a meaningful fit. |
| Poor centerline/fit | Review each origin, angle, bounds, threshold, and overlays. Automatic origins require visual inspection. |
| Excel layout error | Check unique IDs in A and exact T:AR headers, or use standalone Fit data export. |
| File in use / access denied | Save and close it in the other application, check permissions, and retry. PLIFMix does not control Excel or discard edits. |
| Saving blocked after edits | Wait for the calculation matching the new inputs to finish. |
| Slow fit or caching | Processing can take time. Caching needs memory for the folder; reduce worker count if memory is constrained. An FPS entry is a target, not a guarantee. |

For reproducibility, retain numeric inputs, software release/commit, Python/dependency versions, parameters, and numeric exports. A pseudocolor image alone is insufficient. Keep camera gains, geometric/physical calibration, and intensity-to-mass assumptions with the experiment; liquid density differences are not inferred from IDs.

Back up sources and important workbooks. Exports can replace same-named files, and a multi-file operation may save earlier items before a later failure. Resolve the error and inspect the output paths before repeating it. When asking for help, include the page, exact error, environment, and parameters, without uploading private raw data by default.
