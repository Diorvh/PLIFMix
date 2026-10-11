# PLIFMix

[中文说明](README.md)

**Current version: v0.1 demo release.**

PLIFMix is a Windows desktop application for processing planar laser-induced fluorescence (PLIF) images of impinging jets. It supports image averaging, image-intensifier correction, external DaVis processing, mixing analysis, and spatial intensity-distribution fitting.

**Start with the included example:** a fresh download opens the **5 Distribution / 5 质量分布拟合** page and displays three processed sample images automatically. No raw image sequence, Excel workbook, or DaVis installation is needed to try the fitting workflow.

## 1. Install and Start

Use **64-bit Python 3.11 or later with Tcl/Tk** on Windows. Download the repository ZIP and extract the complete folder to a writable location, or clone the repository. Do not run it from inside the ZIP.

Open a terminal in the extracted project folder:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Alternatively, double-click `安装依赖.bat` to install dependencies, then `启动PLIFMix.bat` to launch. Installation and startup must use the same Python interpreter. In VS Code, open the project folder, select that interpreter, and run `main.py`.

Keep the full folder together, including `样例数据`. A downloaded folder name such as `PLIFMix-main` is supported.

## 2. Fit the Included Images

On first launch, the distribution-fitting page (**5 Distribution**) loads these files from `样例数据`:

| Input | Experiment ID | File |
| --- | --- | --- |
| F-only | 67 | `样例数据/实验编号67.npy` |
| Combined FO | 57 | `样例数据/实验编号57.npy` |
| O-only | 45 | `样例数据/实验编号45.npy` |

The example is **DR = 1.3, U = 25**. The supplied image files total approximately 26 MiB; processing parameters are recorded in `样例数据/parameters.json`.

1. Wait for the three original images to appear. Their IDs, origins, angles, and fitting bounds are already populated.
2. Use the button at the upper right to switch between **English and Chinese**. The selection is saved immediately and restored next time.
3. Select **Fit / 拟合** in the image-data row. Wait for the fit to finish; the result table contains transverse, axial, pure elliptical Student, and compact-support fits.
4. Enable **Straighten / 掰直** to view the fitted distribution in straightened coordinates. Select **Original / 原图** or **|Fit - original| / 拟合-原图绝对值** to compare the source and fitted images. The fit view uses the compact-support model when available.
5. Adjust gain or enable Auto for display brightness. Press Enter after editing numerical fields. Changes to processing inputs may require a new fit; returning to Original alone retains the fitted-result cache.

This page normalizes each thresholded image independently.

### Save results

- **One image:** choose **Image output folder / 图像保存目录**, then use **Save / 保存** above the F, O, or FO image. This saves the rendered PNG, including the current view and overlays.
- **Images and fitting data:** select F/O/FO and/or **Fit data / 拟合数据** in **Save selected / 一键保存**, then click its button. Fit data creates a standalone Excel workbook and fit-curve images.
- **An existing experiment table:** select it in **Results workbook / 拟合结果写入文件**, then click **Write fit results / 写入拟合结果**. This requires the column layout described in the detailed guide; no such workbook is needed for the sample.

Default sample output locations are `outputs/sample/images` for individual images and `outputs/sample/distribution` for fitted-result exports. Use distinct output names to avoid replacing previous results.

## 3. Inspect Mixing or Use Your Own Data

Open **4 Mixing / 4 手动调参** to inspect the same sample's mixing-fraction maps and efficiencies. The supplied global O/F is **1.28**. This value is used by mixing analysis, not by the Distribution page.

For your own final processed images, select their folder and enter the F, FO, and O IDs. Files must follow the `实验编号<ID>.npy` naming convention and contain same-sized, finite, two-dimensional arrays. Review the threshold, origins, shifts, angles, and fitting bounds for each new condition; sample parameters are not universal defaults.

For processing from raw frames, follow the full workflow:

| Page | Action |
| --- | --- |
| **1 Average / 平均图** | Select raw frame folders, a range, and a noise threshold; calculate averaged images. |
| **2 Intensifier / 像增强器** | Average calibration frames, fit correction matrices, and apply them to experimental averages. |
| **3 Davis I/O / Davis 整理** | Prepare TIFF inputs for correcting the tilted field of view in external DaVis, then import its exported images. |
| **4 Mixing / 手动调参** | Adjust alignment, inspect mixing maps and efficiencies, and export selected results. |
| **5 Distribution / 质量分布拟合** | Set image origins and centerlines, fit intensity distributions, and export results. |

Mixing and Distribution both read final processed images; fitting does not require PNGs exported from Mixing. All normal pages remain available. The included example contains only final images, not raw acquisition or calibration sequences, and cannot reproduce the upstream workflow by itself. DaVis is separate software and is not included.

The additional tools provide an image viewer and histogram, a cached image player with pixel-history plots, noise analysis, NPY operations, color GIF export, colorbar generation, and comparison of standard distributions.

## 4. Settings and Precautions

- After normal exit, the next launch restores your last page, paths, and parameters from `runtime/settings.json`. Language changes are saved immediately. User-selected folders are not replaced by the sample on every launch.
- The bundled sample paths are portable with the project. External data paths may need to be selected again on another computer.
- Work on copies of raw images: averaging can rename source frames into a numbered sequence.
- Keep floating-point NPY arrays for quantitative processing. Rendered PNGs and GIFs are display exports, not replacements for the input arrays.
- Save and close Excel workbooks before writing to them. The application reports access errors rather than closing Excel or discarding edits.
- Retain the software release or commit ID, source files, settings, and numerical exports together when reporting results in a paper.

If startup reports a missing module, install requirements using the same interpreter used to launch the application. A Tcl/Tk error requires a Python installation with Tk support. Full GUI operation outside Windows has not been established.

## 5. Licenses

- **Code and user documentation:** [MIT License](LICENSE), copyright Lihan Fei.
- **Sample data:** the images and accompanying parameters in `样例数据` are licensed under [CC BY 4.0](样例数据/LICENSE.txt), not the code's MIT License. When reusing them, credit Lihan Fei, link to the PLIFMix repository and the license, and indicate any modifications.
- Third-party dependencies retain their respective licenses.

## Detailed Documentation

See **[the detailed user guide](docs/USER_GUIDE_EN.md)** at **`docs/USER_GUIDE_EN.md`** for page-by-page controls, input organization, interval formats, Excel column requirements, output naming, and troubleshooting.
