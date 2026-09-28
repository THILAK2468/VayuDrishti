# VayuDrishti CPU-First Fine-Tuning Diagnostic Report: CPU_FT_01

## 1. Hardware Environment & Execution Profile
- **CPU Model**: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD
- **Physical Cores**: 6 | **Logical Cores**: 12
- **PyTorch Configured Threads**: 6
- **Available RAM**: 3.38 GB (Total: 15.19 GB)
- **Model Parameter Count**: 5,904,104
- **Model Size**: 22.52 MB (< 30 MB threshold)
- **Image Resolution**: 160x160
- **Batch Size**: 8
- **Training Duration**: 468.1s (7.80 minutes)
- **Single Sample Inference Latency**: 118.99 ms
- **Best Convergence Epoch**: 3

## 2. Definitive Evaluation Metrics

| Split | Storm(s) | Accuracy | Macro F1 | Balanced Accuracy | Weighted F1 | +1h Track Error | +1h Wind MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | BIPARJOY, DITWAH, MOCHA, REMAL, TAUKTAE | 65.58% | 0.6514 | N/A | N/A | N/A | N/A |
| **Validation** | ASANI (Held-out storm) | 74.58% | 0.7094 | 71.29% | 0.7421 | 29.09 km | 1.82 kt |
| **Test** | ASNA (Held-out storm) | 22.54% | 0.1021 | 11.87% | 0.2067 | 59.51 km | 1.46 kt |

## 3. Multi-Horizon Forecasting Breakdown

| Horizon | Validation Track Error (km) | Test Track Error (km) | Validation Wind MAE (kt) | Test Wind MAE (kt) |
| :---: | :---: | :---: | :---: | :---: |
| **+1h** | 29.09 km | 59.51 km | 1.82 kt | 1.46 kt |
| **+3h** | 128.99 km | 157.09 km | 4.5 kt | 2.37 kt |
| **+6h** | 254.92 km | 229.11 km | 8.19 kt | 3.15 kt |

## 4. Diagnostics & Scientific Findings
- **Storm-Level Distribution Shift**: Training storms encompass major Category 3-5 equivalents (`VERY` / VSCS and `EXTREME` / ESCS), whereas Validation (`ASANI`) reached only `SEVERE` (Class 4) and Test (`ASNA`) never exceeded `DEEP_DEPRESSION` (Class 2). Consequently, unrepresented high-intensity classes naturally depress unweighted macro metrics.
- **Overfitting Status**: HEALTHY - No runaway divergence between training and validation trajectories.
- **CPU Execution Efficiency**: Precomputed tensor caching reduced per-epoch overhead to under 1.5 seconds, delivering a **468.1s** total execution time on physical CPU.
