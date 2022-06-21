# hachiroko202206-neuralNetworkEstimation
This repository for the paper "Estimating water quality through neural networks using Terra ASTER data, water depth, and temperature of Lake Hachiroko, Japan (authors: Kai MATSUI, Hikaru SHIRAI, Hiroshi YOKOYAMA, Miyuki ASANO, Yoichi KAGEYAMA)."
This repository was created by Kai Matsui. If you have any question about this repository, you can contact me via email kmatsui19006@gmail.com.

## Author's experimental environment

### PC

- OS: OS: Windows 10 Pro
- CPU: Intel(R) Core(TM) i7-9700F 3.00 GHz
- メモリ: 16.00 GB
- GPU: NVIDIA GeForce RTX 2060

### Tools

- Anaconda: 1.10.0
- Spyder: 5.1.5
- Python: 3.7
- CUDA: 10.1.243
- cuDNN: 7.6.5

## Requirements

| name                 | version  |
| -------------------- | -------- |
| matplotlib           | 3.5.0    |
| numpy                | 1.21.2   |
| opencv-python        | 4.5.4.60 |
| openpyxl             | 3.0.10   |
| pandas               | 1.3.4    |
| progressbar2         | 3.55.0   |
| tensorflow           | 2.1.0    |
| tensorflow-base      | 2.1.0    |
| tensorflow-estimator | 2.1.0    |
| tensorflow-gpu       | 2.1.0    |

## Usage

1. Download all python files and sample data from this Github platform.
2. Open "Run.py" in the Python IDE (such as Anaconda Spyder) on your computer.
3. Run the file 'run.py'
4. Check the results in the *dst* directory
   - nn_results - pred_results.pkl: 
   - evaluate_proposed.xlsx: 
   - water_quality_map.png: 

## Sample data

Dataset C, near infra-red (Band 3)

water quality: Suspended Solid (SS)

evaluate_summary.xlsx: This file were summarized data: estimated values and errors of the proposed method and fuzzy regression analysis; actual SS values at five measurement sites.
