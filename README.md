# PLIFMix

PLIFMix is a desktop application for processing planar laser-induced fluorescence (PLIF) images of impinging jets. It provides image averaging, image-intensifier correction, file preparation for external DaVis processing, mixing analysis, and spatial intensity-distribution fitting.

The interface supports **English and Chinese**. Use the language button in the upper-right corner; file names, experiment IDs, and numerical inputs remain unchanged.

## Install and Run

Use **64-bit Python 3.11 or later with Tcl/Tk** on Windows. Download or clone the repository, extract it to a writable folder, and open a terminal in that folder:

```powershell
python -m pip install -r requirements.txt
python main.py
```

The folder may keep its downloaded name, such as `PLIFMix-main`. Alternatively, double-click `安装依赖.bat` to install dependencies and `启动PLIFMix.bat` to start the application. Dependency installation and startup must use the same Python interpreter.

DaVis is a separate application and is not included. It is not needed for the bundled final-image example. Full GUI operation on other operating systems has not been established.

## Processing Workflow

| Step | Page | What to do |
| --- | --- | --- |
| 1 | **Average** | Select raw frame folders, specify the frame range and noise threshold, and calculate an average for each experiment. |
| 2 | **Intensifier** | Average calibration frames, fit and save a matrix for each gain, then correct the experimental averages. |
| 3 | **Davis I/O** | Prepare gain-scaled TIFFs, perform the geometric correction in external DaVis, and import its exported images. |
| 4 | **Mixing** | Select F/FO/O images, adjust alignment and analysis parameters, inspect mixing results, and export images or curve data. |
| 5 | **Distribution** | Select the final images, inspect origins and centerlines, fit the distributions, and export images and fitted parameters. |

Steps 4 and 5 both read the final processed images. Step 5 does not require the display images exported from step 4.

**[Read the user guide](docs/USER_GUIDE.md)** for page-by-page instructions, input formats, save locations, and troubleshooting. Chinese control names are included alongside English labels.

## Included Example

The `样例数据` folder contains three final processed NPY images, approximately 26 MiB in total:

| Role | ID | File |
| --- | --- | --- |
| F | 67 | `样例数据/实验编号67.npy` |
| FO | 57 | `样例数据/实验编号57.npy` |
| O | 45 | `样例数据/实验编号45.npy` |

The example is labeled **DR = 1.3**, **U = 25**, with **global O/F = 1.2786069651741294**. Its processing inputs are stored in `样例数据/parameters.json`.

The bundled group preset contains only this example. To process other experiment groups, select your own preset file or enter the IDs and conditions manually.

1. Start the application and switch to English if needed.
2. Open **4 Mixing**. Unconfigured image folders default to the example; previously selected user folders are preserved. Check F = 67, FO = 57, and O = 45.
3. Inspect the mixing maps and calibrated/uncalibrated efficiency rows. Press Enter after changing an input.
4. Open **5 Distribution**, check the same IDs, and select **Fit**. Wait for the fitted parameters and images.
5. Select the export items and use **Save selected**, or save an individual image from its panel.

No raw acquisition sequences, calibration sequences, or DaVis input files are included. The example covers downstream mixing and distribution fitting, not reconstruction of the entire experiment from raw frames.

## Other Tools

| Tool | Purpose |
| --- | --- |
| Image viewer | Inspect pixels and a histogram; adjust the display and export an image or numbered TIFF sequence. |
| Image player | Cache and inspect frames; plot the intensity history at a selected pixel. |
| Noise analysis | Inspect frame-maximum trends to help choose an absolute threshold. |
| NPY tools | Inspect arrays, list their largest values, and save simple transformations. |
| GIF export | Export a numbered frame range as a color GIF using source color channels. |
| Colorbar | Generate a labeled colorbar image. |
| Distributions | Compare standard symmetric distribution shapes. |

## Before Processing Your Own Data

- **Use working copies of raw images.** Averaging checks the sequence and may rename source files to `0000`, `0001`, and so on.
- Retain floating-point **NPY files** as processing inputs. Rendered PNGs, GIFs, and gain-scaled TIFFs are not substitutes for these arrays.
- Use separate output directories and distinct names for results you need to retain. Repeated exports can replace same-named files.
- Save and close an Excel workbook before writing results into it. PLIFMix reports access errors; it does not close Excel or discard unsaved edits.
- Parameters and the last page are remembered in `runtime/settings.json`. Supplied values are example settings, not universal processing defaults.

Retain the software version or commit ID, input files, parameter settings, and numerical exports together. To report a problem, include the page, error message, operating system, Python version, and relevant parameters; do not attach private datasets by default.
