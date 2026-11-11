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
hf auth login                  # paste a Hugging Face token that can read the dataset (ask the owner)
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
python run.py --method stcgn     --setting irrg   # automatically runs the St-CGN source warm-up (2000 updates on source chips) first, then adapts
python run.py --method picst     --setting rgb
```

- Outputs: `runs/<setting>/<experiment>/seed1/report/PEAK_AND_ENDPOINT.md`.
- Re-running the same command resumes from the saved state.
- `--hours H` pauses safely at the next evaluation after H hours (e.g. before a Kaggle session limit).
- Evaluate a checkpoint:

```bash
python run.py --method eval --setting irrg --checkpoint <path/to/checkpoint.pth>
```
