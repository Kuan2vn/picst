# Results so far (seed 1, 10/11/2026)

All rows use the protocol in the README. Each cell gives the mIoU6 / mIoU5 peak, selected with target-eval labels (disclosed), then (iteration); "last" is the actual final evaluation.
- **Platform:** Kaggle T4 unless marked HPC (A5000). Cross-platform rows are not a matched comparison.
- **Seeds:** single seed. Multi-seed runs come after the table is fixed.

## VH IRRG → PD IRRG file (main)

| Method | Peak mIoU6 / mIoU5 (iter) | Last mIoU6 (iter) |
|---|---|---|
| Source only | 51.97 / 61.74 (0) | — |
| DT-ST (CVPR'23) | 58.17 / 69.53 (7k) | 58.17 (9k) |
| Stable-ST (IJCV'26) | 57.14 / 68.28 (6k) | 56.73 (8k) |
| SND (CVPR'24), author pipeline | pending | |
| S3T-SFDA (TGRS) | pending | |
| St-CGN (TGRS'26) | pending | |
| **PIC-ST v1 (ours)** | **60.94 / 72.86 (5k)** | 60.49 (7k) |
| PIC-ST revised (ours, `picst`) | running | |

PIC-ST ablations (IRRG):

| Arm | Peak (iter) | Last (iter) |
|---|---|---|
| C0 only | 60.31 (4k) | 57.37 (6k) |
| C1 only | 57.64 (6k) | 57.23 (8k) |
| C0 + C1 | 60.94 (5k) | 60.49 (7k) |

## VH IRRG → PD RGB

| Method | Peak mIoU6 / mIoU5 (iter) | Last mIoU6 (iter) |
|---|---|---|
| Source only | 39.78 / 47.09 (0) | — |
| DT-ST | 39.78 (0) | 39.24 (2k) |
| PIC-ST v1 | 43.73 / 52.21 (4k) | 43.22 (6k) |
| PIC-ST C0 (tree) | 48.06 / 57.43 (6k) | 46.46 (8k) |
| **PIC-ST revised (HPC)** | **50.26 / 60.12 (3k)** | 47.27 (5k) |
| per-class C0 only (HPC) | 50.35 / 60.23 (3k) | 48.17 (5k) |

The revised PIC-ST was designed after inspecting RGB, so RGB is a development setting for it. Generality must be shown on settings not used to design it.
