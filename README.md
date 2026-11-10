# PIC-ST and SFDA baselines for VH → PD remote-sensing segmentation

Source-free domain adaptation (SFDA) for 6-class semantic segmentation, from **ISPRS Vaihingen (VH, IRRG)** to
**ISPRS Potsdam (PD)**. Every method starts from the same SegFormer-B2 source model, uses the same data and
the same training and evaluation protocol, and runs from one launcher.

| Setting | Source → target | Role |
|---|---|---|
| `irrg` | VH IRRG → PD IRRG file (official `3_Ortho_IRRG`, bands IR,G,B) | main table |
| `rgb` | VH IRRG → PD RGB (`2_Ortho_RGB`) | second setting (standard SFDA benchmark) |

## Methods

| `--method` | Method | Venue | Code basis |
|---|---|---|---|
| `picst` | **Ours (PIC-ST, revised)**: DT-ST + per-class gate G + C0 + plausible C1 | — | this repo |
| `picst_v1`, `picst_c0`, `picst_c1` | PIC-ST first version and ablations | — | this repo |
| `dtst` | DT-ST (host) | CVPR 2023 | author code (B2 port) |
| `snd` | SND (author pipeline: source weights + SND-warm banks) | CVPR 2024 | author code (B2 port) |
| `stable_st` | Stable-ST (NFD dynamic teacher + bi-level denoising) | IJCV 2026 | author code (B2 port) |
| `s3t_sfda` | S3T-SFDA (spatial views, historical agreement, historical contrast) | IEEE TGRS | re-implemented from paper + released code |
| `stcgn` | St-CGN (MsFCL source warm-up, CbPCM, DvPEF) | IEEE TGRS 2026 | re-implemented from paper (no public code) |

These are **re-implementations on a common protocol, not reproductions** of the published numbers. What
follows each paper/code and what differs is listed in [docs/METHODS.md](docs/METHODS.md). Current seed-1 results
are in [docs/RESULTS.md](docs/RESULTS.md).

## 1. Install (Linux, one NVIDIA GPU with ≥ 12 GB free)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 2. Data (private Hugging Face dataset)

Ask the owner for read access to `kuan2/picst-rs-sfda-data`, then:

```bash
huggingface-cli login
python download_data.py            # about 6.6 GB; fills ./data
```

`./data` then contains:
- `gsd_sfuda_b2_irrg_v1/`: PD IRRG-file target (3456 unlabelled train / 2016 labelled eval chips, 512 px) and the labelled VH source chips;
- `siamseg_sfuda_b2_v1/`: the PD RGB target (same chips and labels);
- `source_40000.pth`: the shared source model;
- `snd_warm/`: the SND warm stage (stable-sample banks and warm protocol), used by `--method snd`.

The data derive from the ISPRS 2D Semantic Labeling benchmark. Use them for research only, under the ISPRS terms.

## 3. Run

```bash
python run.py --method picst --setting irrg        # ours, main setting
python run.py --method dtst  --setting irrg        # host
python run.py --method stcgn --setting irrg        # runs the St-CGN source warm-up first (~2000 updates)
python run.py --method picst --setting rgb         # ours on VH IRRG -> PD RGB
```

- Outputs go to `runs/<setting>/<experiment>/seed1/`. One run takes about 3–8 h on a T4/A5000 class GPU.
- Each run writes `report/PEAK_AND_ENDPOINT.md`: mIoU6 and mIoU5 at the selected peak, and at the actual last evaluation, with per-class IoU.
- `RUN_STATE.json` shows the status (`finished` or `paused_for_next_session`).
- Re-running the same command resumes from the saved full state.
- To evaluate any checkpoint:

```bash
python run.py --method eval --setting irrg --checkpoint runs/irrg/pheno_c0_c1plaus/seed1/warm/best_target_val.pth
```

## 4. Protocol (identical for every method)

- **Model and initialisation:** SegFormer MiT-B2 + MLP decoder, initialised from `source_40000.pth`.
- **Training:**
  - batch 6, crop 512, seed 1;
  - AdamW 2e-5, decoder ×10, readout ×1, weight decay 0.01;
  - linear warm-up of 1500 updates, then linear decay over a 32000-update horizon.
- **Evaluation:** every 1000 updates, on all 2016 target-eval chips, by the EMA teacher. Metrics are mIoU6 (all classes) and mIoU5 (clutter excluded).
- **Stopping:**
  - stop after 2 evaluations without a strictly higher mIoU6 peak;
  - cap at 9000 updates; if the cap evaluation is itself a new peak, continue until the first decline (hard maximum 20000).
- **Reporting:** the peak (selected with target-eval labels, disclosed), metrics from that same checkpoint, and the actual last evaluation.
- **Target-train labels** are never read.

## 5. Tests

```bash
cd code && python -m unittest discover -s tests -p "test_rs_baselines.py" -v
cd code && python -m unittest discover -s tests -p "test_*kaggle*.py"
```

## Code map

| Path | Content |
|---|---|
| `code/train_SND.py`, `code/train_SND_remote.py`, `code/train_DTST_remote.py` | Shared B2 trainer (DT-ST/SND family) with opt-in hooks |
| `code/scripts/dtst_phenology.py` | PIC-ST C0 / C1 (gate decisions in `kaggle_settings/pheno_*.json`) |
| `code/scripts/rs_baselines.py` | S3T-SFDA and St-CGN adaptation losses |
| `code/train_stcgn_warmup.py` | St-CGN source warm-up and anchor prototypes |
| `code/scripts/dtst_remote.py` | DT-ST teacher feedback and the Stable-ST NFD proxy |
| `code/scripts/run_dtst_kaggle.py`, `code/scripts/run_baseline_kaggle.py` | Runners: stopping rule, checkpoint-safe pause/resume, reports |
| `code/kaggle_settings/` | All recipes and method settings (one JSON per arm) |

Third-party code: the SegFormer model code follows NVIDIA SegFormer (see `code/references/segformer/LICENSE`, non-commercial). The DT-ST/SND trainer follows DZhaoXd/DT-ST and DZhaoXd/SND.
