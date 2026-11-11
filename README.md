# PIC-ST and SFDA baselines (VH → PD)

Internal copy for the authors. Source-free domain adaptation from ISPRS Vaihingen (IRRG) to Potsdam, SegFormer-B2.
Method details: [docs/METHODS.md](docs/METHODS.md). Results: [docs/RESULTS.md](docs/RESULTS.md).

## 1. Install (Linux, one NVIDIA GPU with ≥ 14 GB free; a 16 GB T4 works)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 2. Setup data (private Hugging Face dataset `kuan2/picst-rs-sfda-data`, ~6.6 GB)

```bash
huggingface-cli login
python download_data.py        # fills ./data
```

## 3. Run

`--setting` is `irrg` (PD IRRG, main) or `rgb` (PD RGB). Methods: `picst`, `picst_v1`, `picst_c0`, `picst_c1`, `dtst`, `snd`, `stable_st`, `s3t_sfda`, `stcgn`.

```bash
python run.py --method picst     --setting irrg   # ours
python run.py --method dtst      --setting irrg
python run.py --method snd       --setting irrg
python run.py --method stable_st --setting irrg
python run.py --method s3t_sfda  --setting irrg
python run.py --method stcgn     --setting irrg   # runs the St-CGN source warm-up first (see below)
python run.py --method picst     --setting rgb
```

- Outputs: `runs/<setting>/<experiment>/seed1/report/PEAK_AND_ENDPOINT.md`.
- Re-running the same command resumes from the saved state.
- `--hours H` pauses safely at the next evaluation after H hours (e.g. before a Kaggle session limit).
- Evaluate a checkpoint:

```bash
python run.py --method eval --setting irrg --checkpoint <path/to/checkpoint.pth>
```

## St-CGN: what the "source warm-up" is

`--method stcgn` is a two-stage run, done automatically by the same command (`code/train_stcgn_warmup.py`, then the normal adaptation):

1. **Warm-up (source side, labelled VH source chips, 2000 updates).** Starts from the shared `source_40000.pth`, which plays the role of the paper's pre-trained teacher. The student is trained with `L_src + L_inst/L_reg (contrastive, weighted 0.3/0.7) + L_kpd + L_kfd` (cross-entropy on source labels, prototype/region contrastive losses between two augmented views, and KD from an EMA teacher on predictions and decoder features). Output in `runs/irrg/<exp>/seed1/warmup/`:
   - `stcgn_warmup.pth`: EMA-teacher weights, used as the **initial model of the adaptation stage** (instead of `source_40000.pth`);
   - `stcgn_prototypes.pt`: per-class anchor prototypes (class-masked mean decoder features over all source chips), used to initialise the CbPCM prototypes.
2. **Adaptation** (target PD, same protocol as every other method) starting from `stcgn_warmup.pth`.

The warm-up is skipped on re-run if `warmup/WARMUP_COMPLETE.json` exists and its checksums match. Fixed hyper-parameters not given in the paper are listed in the header of `code/train_stcgn_warmup.py`.
