# Methods, settings and re-implementation notes

All methods share the protocol in the README. The rows below list exactly what each method adds to the shared
trainer, and every point where a re-implementation had to choose or differs from the paper or code.

## Ours: PIC-ST (phenology-interventional self-training), revised

PIC-ST = DT-ST + G + C0 + C1. Code: `code/scripts/dtst_phenology.py`; arms: `code/kaggle_settings/pheno_*.json`.

### G: dev-forecast gate, decided once before training

- **Candidates.** The vegetation classes are declared a priori: tree and low vegetation.
- **Forecast.** For each candidate, a label-free δ̂ is computed: the median margin drop under the leaf-off counterfactual x' = μ_ground + 0.5 (x − μ_class). The test-time gain at δ̂ is then forecast on the labelled dev tiles. Labels never enter training.
- **Rule.** A class is enabled when its own forecast gain is ≥ +1 mIoU, provided the joint forecast is ≥ the best single class.

| Setting | tree | low vegetation | Enabled |
|---|---|---|---|
| IRRG | +1.43 | −15.39 | tree only |
| RGB | +2.59 | +1.52 (joint +3.44) | tree + low vegetation |

### C0: teacher-side logit compensation

- Each enabled class gets δ̂ added to its teacher logit before pseudo-labelling.
- δ̂ is re-estimated every 50 updates, pooled over 20 batches.

### C1: student-side counterfactual

- C1 applies the leaf-off counterfactual (probability 0.5, a ~ U[0.25, 1]) inside teacher-tree regions of the student input; the label is unchanged.
- **Plausible C1 (revision):** an edited pixel is kept only where the teacher still predicts a vegetation class on the edited input.

### Arms (RGB uses `pheno_c0multi*`, IRRG uses tree-only C0)

| Arm | C0 | C1 |
|---|---|---|
| `pheno_c0_c1plaus` | tree | plausible |
| `pheno_c0multi_c1plaus` | tree + low vegetation | plausible |
| `pheno_c0_c1` (v1) | tree | original |
| `pheno_c0`, `pheno_c0multi`, `pheno_c1`, `pheno_null` | ablations | |

`pheno_null` is bit-identical to DT-ST.

## DT-ST (Zhao et al., CVPR 2023): host

- Dynamic teacher update with the soft-ND proxy: query start 9, step 3, max interval 30, EMA 0.99.
- TCR candidate resampling with copy-paste (top-k 0.5, refresh 3000).
- Negative learning, weight 0.5.

## SND (Zhao et al., CVPR 2024), `--method snd`

Follows the author's `run.sh`:
1. A warm stage (fixed-EMA self-training: EMA 0.99 every 12 updates, no meta, TCR) builds the stable-sample banks. The warm-stage weights are **not** reused.
2. The SND stage restarts from the **source** weights, with bi-level meta pseudo-label weighting (top-k 0.01, refresh 3000), EMA 0.999 every update and the LR clock from 0.

**Deviation.** The provided warm stage (`data/snd_warm`) ran 20k updates with the older readout multiplier ×10. It only provides banks.

## Stable-ST (Zhao et al., IJCV 2026), `--method stable_st`

The author `train_stable_st.py` is the SND trainer plus an NFD (neighbour feature density) dynamic-teacher proxy. This repo uses the author's full configuration (`deeplabv2_r101_StableST_G2C.yaml`):
- NFD dynamic teacher: query start 10, step 5, max interval 21, EMA 0.99;
- bi-level weighting: top-k 0.05, refresh 3000;
- TCR;
- no negative learning.

**Deviations:**
1. The author SegFormer config disables DTU and meta-weighting, because their NFD code needs a tensor feature. Here NFD is computed on the last MiT stage.
2. The released line `masked_fill(index_tensor, -1)` is not a valid mask, so the intended self-exclusion is implemented.
3. The optimiser and budget are the common protocol.

## S3T-SFDA (Xi et al., IEEE TGRS), `--method s3t_sfda`

Re-implemented from the paper and the released code (github.com/yixinzs/S3T-SFDA, `MVCSFDAEncoderDecoder`). Where the two disagree, the concrete released code is followed and the difference is listed.

| Component | Implemented as (released code) | Paper / note |
|---|---|---|
| Host | DT-ST dynamic teacher (soft-ND; 9/3/30; EMA 0.99), TCR paste, positive labels with running confidence, negative BCE **weight 1** | same |
| SMVC | With probability 0.5 per update: 4 square ROIs per image, side ∈ [0.5, 1] × crop, `roi_align` to 256²; teacher pseudo-labels and negative targets `roi_pool`-ed to the same ROIs | The paper averages teacher predictions over N views. The released code crops one full-image teacher prediction, which is consistent by construction |
| TDC | Historical model = student snapshot every 300 updates; CE weight 2 where the historical argmax agrees with the pseudo-label, else 0.5 | The paper uses w = 1 − sigmoid(KL(current, oldest)). That form is not in the released loss, so it is not used |
| STC | Conv-MLP projection head (1024 → 1024 → 256, BN, ReLU, L2) on the 4 concatenated MiT stages at 1/8; historical-feature contrast (τ 0.07), positives = same pseudo-label, reliability from historical entropy clamped to [0.9, 1.1]; weight 0.5; head LR ×10 | same |
| Entropy L_E | not used | In the paper, absent from the released loss |
| Teacher EMA | whole network | The released code moves only the decode head; the paper's Eq. 3 is the whole network |

## St-CGN (Zhang et al., IEEE TGRS 64, 2026), `--method stcgn`

No public code; re-implemented from the paper. Two stages.

### 1. Source warm-up (`code/train_stcgn_warmup.py`)

- Runs before adaptation on the labelled VH source, starting from the shared source40k, which replaces the paper's 3000-update teacher pre-training.
- 2000 updates of L_src + λ_inst L_inst + (1 − λ_inst) L_reg + L_kpd + L_kfd.
  - **L_inst:** class-masked prototypes of two augmented views; InfoNCE over the classes present in both.
  - **L_reg:** region matching of teacher anchors to the most similar student pixel, with a curriculum threshold ε₀ = 0.7, ε_max = 0.9 and 255 negatives.
  - **L_kpd / L_kfd:** teacher → student KL on predictions and MSE on decoder features.
- Teacher EMA 0.996.
- The source anchors (Eq. 1) are computed from the teacher's decoder features.
- **Chosen where the paper is silent:**
  - λ_inst = 0.3 (the paper says lower instance weight is better);
  - InfoNCE τ = 0.1;
  - 256 anchors per image;
  - projection MLP 768 → 768 → 256;
  - KD weights 1;
  - source optimiser recipe (AdamW 6e-5, head ×10, linear decay), batch 6.

### 2. ST stage (`code/scripts/rs_baselines.py`, class `StCGN`)

- **Teacher and inputs.** Mean teacher, EMA 0.996 every update. There is no dynamic teacher, no negative learning and no copy-paste: the student sees the unpasted strong view.
- **CbPCM.** The rectified distribution is P̂ = norm(P ⊙ softmax(cos(z, p_k) / 0.1)), where P is the student's no-grad prediction on the weak view, as in the paper. The label is argmax P̂, kept when the original P of that class reaches the class threshold: the top-p_ratio value of P among that class's predictions in the mini-batch. p_ratio rises from 0.4 to 0.9 over 5000 updates.
- **DvPEF.** L_kld = KL(P_t ‖ P_s), plus the marginal-entropy term on (P_t + P_s)/2. Weights: λ_ent 0.1, λ_kld 0.9.
- **Prototype refresh.** Every 1000 updates, prototypes become the mean target feature of accepted pseudo-labels. The paper says "updated every 1000 iterations" without the rule.
- **Not used:** ASGCF, a DeepLabV3+ decoder module. All methods keep the same B2 architecture.
