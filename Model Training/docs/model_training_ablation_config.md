# Model Training — Ablation Config & Process Reference

Living reference for the crack-segmentation training pipeline: what was fixed,
what's being compared, where everything lives, and what to do once each run
finishes. Update the "Status / Results" column as runs complete.

---

## 1. Background — the polarity bug (why any of this mattered)

OmniCrack30k masks store the crack as the **dark** class (background = white);
CrackSeg9k stores it the opposite way (crack = white). The original converter
assumed one polarity for both datasets, so **all 30,017 OmniCrack30k labels
became full-frame "crack" polygons** — poisoning ~80% of the training set with
"the entire image is a crack." Fixed with a per-dataset `invert` flag in
`convert_annotations.py` (commit `a07d863`).

Two follow-on fixes to the conversion itself:
- **Dilation → pixel-count filter** (`2cab145`): initial fix dilated masks to
  recover thin cracks the area-filter was dropping, but that inflated crack
  width ~2px — bad for downstream length/width measurement. Replaced with a
  **pixel-count filter** (not `cv2.contourArea`) so 1px hairlines survive at
  **true width**, no dilation. `MIN_CRACK_PIXELS = 15` (tuned from a
  miss-rate/speck-size sweep, `480c818`).
- **mAP is the wrong metric for this task** (see §4) — a 1px offset on a
  hairline crack drops instance IoU below the 0.5 gate, scoring a correctly
  found crack as a total miss. Field-correct pixel metrics adopted instead.

---

## 2. Dataset & pipeline

| dataset | raw images | true background | source |
|---|---|---|---|
| CrackSeg9k | 7,332 | ~15.6% | HF `rimvydasrub/crackseg9k` |
| OmniCrack30k | 30,017 | ~10.9–16.5% (by split) | HF benchmark release |
| **Total** | **37,349** | | |

**Split** (`make_subset.py --all --link`): OmniCrack30k's **official**
train/val/test preserved (no reshuffle — prevents near-duplicate frame
leakage across splits); CrackSeg9k has no official split, so it's randomly
split 70/15/15.

| split | count |
|---|---|
| train | 27,290 |
| val | 4,376 |
| test | 5,683 |

Splits are **symlinks** (`--link`) into `data/raw/`, not copies — near-instant
to build, no duplicated storage on the network volume.

**Conversion params** (`src/config.py`): `MIN_CRACK_PIXELS = 15`,
`EPSILON_RATIO = 0.0005`, no dilation. Verified: 0% (CrackSeg9k) / 1.2–2.5%
(OmniCrack30k, by split) miss rate — remaining misses are sub-15px specks
below measurable resolution.

---

## 3. Pipeline commands (reproducibility)

```bash
# 1. Convert masks -> YOLO polygon labels (per-dataset polarity handled internally)
python src/convert_annotations.py

# 2. Build the split (symlinked, official OmniCrack30k splits preserved)
python src/make_subset.py --all --link --sources \
  "data/raw/crackseg9k/images:data/yolo/labels_raw/crackseg9k:auto" \
  "data/raw/omnicrack30k/images/training:data/yolo/labels_raw/omnicrack30k/training:train" \
  "data/raw/omnicrack30k/images/validation:data/yolo/labels_raw/omnicrack30k/validation:val" \
  "data/raw/omnicrack30k/images/test:data/yolo/labels_raw/omnicrack30k/test:test"

# 3. Train (per slot, see §5 for --mode / --config values)
python src/train.py --mode <slot>              # YOLO-family (A, B, C)
python src/train_unet.py --config <slot>       # U-Net-family (D, E)

# Resume after an interruption (checkpoints saved every epoch):
python src/train.py --mode <slot> --resume
python src/train_unet.py --config <slot> --resume

# 4. Evaluate on the held-out TEST split (pixel metrics, not mAP — see §4)
python src/eval_segmentation.py --weights <best.pt> --split test   # YOLO models
python src/eval_unet.py --weights <best.pt> --split test           # U-Net models
```

---

## 4. Evaluation methodology

**mAP is not used as the headline metric.** It's an instance metric with a
0.5-IoU pass/fail gate — for a 1px-wide crack, a 1px localization offset
drops instance IoU below 0.5 and scores a correctly-found crack as a total
miss. No crack-segmentation paper in the literature reports mAP; the field
uses pixel-level metrics instead (Segment-Any-Crack: F1/IoU; OmniCrack30k
benchmark/nnU-Net: clIoU₄ₚₓ).

**Metrics computed** (`eval_segmentation.py` for YOLO, `eval_unet.py` for
U-Net — same underlying `Accum` metric math, reused not duplicated):

| metric | measures | tolerant? |
|---|---|---|
| Pixel IoU | strict area overlap | no |
| F1 / Dice | area overlap, TP double-counted | no |
| **clIoU₄ₚₓ** | IoU with 4px tolerance buffer | yes — the OmniCrack30k benchmark metric |
| **clDice** | centerline connectivity (skeleton-based) | topology, not area — critical for length/width measurement fidelity |
| Precision / Recall | pixel-level | — |

Evaluated on the **test** split (never touched during training or
early-stopping), against the **original GT masks** (polarity-corrected), not
the lossy polygon labels — resolved via `resolve_gt()`/`load_gt()`
(`eval_segmentation.py`), reused by `eval_unet.py`. Reported **micro**
(pixel-pooled, primary — matches the literature) and **macro** (per-image
mean, secondary/conservative), across a confidence/threshold sweep, per
dataset (overall / omnicrack30k / crackseg9k).

**Slot B (full-baseline) results** — the only slot evaluated so far
(conf 0.25 default / conf 0.10 best-F1):

| dataset | metric | @0.25 | @0.10 (best-F1) |
|---|---|---|---|
| Overall | F1 | 0.612 | 0.638 |
| Overall | IoU | 0.441 | 0.468 |
| Overall | clIoU₄ₚₓ | 0.568 | 0.617 |
| OmniCrack30k | F1 | 0.593 | 0.622 |
| OmniCrack30k | IoU | 0.421 | 0.451 |
| OmniCrack30k | clIoU₄ₚₓ | 0.538 | 0.589 |
| CrackSeg9k | F1 | 0.717 | 0.729 |
| CrackSeg9k | IoU | 0.559 | 0.573 |
| CrackSeg9k | clIoU₄ₚₓ | 0.759 | 0.792 |

**Literature comparison** (OmniCrack30k, F1/IoU from Segment-Any-Crack paper
`arXiv:2504.14138`; clIoU₄ₚₓ from Benz & Rodehorst, CVPR-W 2024):

| model | F1 | IoU | clIoU₄ₚₓ |
|---|---|---|---|
| **Ours (full-baseline, best-F1)** | **0.622** | **0.451** | 0.589 |
| SAC (SAM-based) | 0.612 | 0.441 | n.r. |
| SegFormer | 0.600 | 0.429 | n.r. |
| DeepLabv3+-ResNet101 | 0.565 | 0.394 | n.r. |
| U-Net (their baseline) | 0.543 | 0.373 | n.r. |
| nnU-Net (heavy ensemble) | n.r. | n.r. | **0.64** |

Small `yolov8s-seg` leads every model that reports F1/IoU; trails only the
heavy nnU-Net ensemble on clIoU. (`n.r.` = not reported in that paper —
different papers report different metric subsets, which is itself the
reason a field-correct multi-metric eval script was built.)

---

## 5. Training slots

Design principle: each slot changes **exactly one variable** from the fixed
baseline (B), so score differences are attributable to that one change.

| slot | script / mode | base model | imgsz | epochs | batch | isolates | output dir |
|---|---|---|---|---|---|---|---|
| **A** | `train.py --mode full` | yolov8s-seg | 1280 | 200 | autobatch | resolution (recall/clIoU fix) | `runs/full` |
| **B** | `train.py --mode full-baseline` | yolov8s-seg | 640 | 150 | 16 | — (fixed reference) | `runs/full_baseline` |
| **C** | `train.py --mode full-yolo11` | yolo11s-seg | 640 | 150 | 16 | architecture generation | `runs/full_yolo11` |
| **D** | `train_unet.py --config unet` | U-Net-ResNet34 | 640 | 150 | 16 | semantic arch + imbalance loss | `runs/full_unet` |
| **E** | `train_unet.py --config unet-1280` | U-Net-ResNet34 | 1280 | 150 | 4 | does A's resolution hypothesis transfer to a decoder without YOLO's 160px proto bottleneck? | `runs/full_unet_1280` |

**Full hyperparameters** — YOLO-family (A/B/C) share: SGD, lr0=0.01,
momentum=0.937, weight_decay=5e-4, warmup=3ep, AMP on, mosaic=1.0
(`close_mosaic=10`), fliplr=0.5, scale=0.5, hsv(h=.015,s=.7,v=.4),
patience=30. U-Net-family (D/E): AdamW, lr=1e-3, weight_decay=1e-4,
Focal Tversky loss (α=0.3, β=0.7, γ=1.33 — β>α weights false negatives
harder, directly targeting the established recall bottleneck), patience=30
(no-improve-epochs on val Dice).

**yolo11s-seg pretrained-weight note**: ultralytics' own asset-resolution
(`get_github_assets`) pointed at release tag `v8.4.0`, but that checkpoint's
pickled `C3k2` class didn't match the installed `ultralytics==8.4.101`
package internals (assets-repo and PyPI-package releases aren't guaranteed
version-locked). Fixed by pinning `ultralytics==8.3.0` — the last version
verified to load its own matching `v8.3.0`-tag checkpoint cleanly.
yolo12s-seg was considered and **rejected**: no pretrained segmentation
checkpoint exists upstream (only detection weights), so it would train from
random init — not a fair same-epoch comparison against a transfer-learned
baseline.

### Status / Results

| slot | status | mask mAP50 (val, training-time) | test-set F1 / IoU / clIoU₄ₚₓ | HF upload |
|---|---|---|---|---|
| A | training (Japan, GPU0) | — | — | — |
| B | **done** | 0.192 (epoch 150) | 0.622 / 0.451 / 0.589 (best-F1) | ✅ `johnnytan5/2nd-run-fyp-train` |
| C | training (Japan, GPU1) | — | — | — |
| D | training (Japan, GPU2) | — | — | — |
| E | training (Japan, GPU3) | — | — | — |

*(Fill in as each completes — see §7 for the exact eval commands.)*

---

## 6. Directory / storage map

Each slot writes to a **completely separate `project` directory** — verified
no collision risk when run in parallel (all 4 running simultaneously on one
GPU box, one process per GPU via `CUDA_VISIBLE_DEVICES`):

```
runs/
├── full/exp/weights/{best,last}.pt              # slot A
├── full_baseline/exp/weights/{best,last}.pt     # slot B (done, on a different machine — L4/RunPod)
├── full_yolo11/exp/weights/{best,last}.pt       # slot C
├── full_unet/exp/weights/{best,last}.pt         # slot D
└── full_unet_1280/exp/weights/{best,last}.pt    # slot E
```

YOLO slots (A/B/C) also get ultralytics' standard run artifacts alongside
`weights/`: `results.csv`, `results.png`, `args.yaml`, `val_batch*_pred.jpg`,
confusion matrix, PR curves, etc. U-Net slots (D/E) only produce
`weights/{best,last}.pt` (`train_unet.py`'s own lightweight checkpointing —
no equivalent of ultralytics' plot suite).

U-Net checkpoints (D/E) also embed their own training `cfg` dict inside the
`.pt` file (`torch.load(...)["cfg"]`) — `eval_unet.py` reads `imgsz` from
this automatically, so the 640px (D) vs 1280px (E) checkpoints can never be
scored with a mismatched resolution by mistake.

---

## 7. Post-training checklist (per slot, once training finishes)

1. **Confirm the run actually finished / didn't silently die**:
   ```bash
   tail -30 train_<X>.log
   ls -la runs/<project>/exp/weights/
   ```
2. **Run the pixel-metric eval on the test split**:
   ```bash
   # YOLO slots (A, C):
   python src/eval_segmentation.py --weights runs/<project>/exp/weights/best.pt --split test

   # U-Net slots (D, E):
   python src/eval_unet.py --weights runs/<project>/exp/weights/best.pt --split test
   ```
   Saves `results/seg_eval_<tag>_test.{json,csv}`.
3. **Fill in the Status/Results table in §5** of this doc with the real
   numbers.
4. **Upload to HuggingFace** (see §8 for repo/naming convention).
5. **Update the literature-comparison table** if a new slot beats/matches
   the current best (currently slot B).

---

## 8. HuggingFace upload

**Existing repos:**
- `johnnytan5/crack-raw-data` (dataset) — raw CrackSeg9k + OmniCrack30k
  tarball, used to migrate data across cloud providers/regions.
- `johnnytan5/2nd-run-fyp-train` (model) — slot B (full-baseline) only, so
  far. Structure:
  ```
  2nd-run-fyp-train/
  ├── best_full-baseline.pt       # renamed from best.pt for clarity (server-side, no re-upload)
  ├── run/                        # full ultralytics run dir (weights/, results.csv/png, args.yaml, curves)
  │   └── config.json             # training config + platform + test metrics snapshot (see below)
  └── eval/                       # seg_eval_test.{json,csv}
  ```

**⚠️ Open decision — not yet made**: slots A/C/D/E need a home. Two options:
- **(a) Same repo, new subfolders** — e.g. `full-1280/`, `full-yolo11/`,
  `unet/`, `unet-1280/`, each mirroring the `best_<slot>.pt` +
  `<slot>/run/` + `<slot>/eval/` pattern above.
- **(b) New repo** for this batch (different infra/session from B — trained
  on Alibaba Japan L20s, not RunPod L4).

Lean toward **(a)** for a single source of truth per model-comparison story,
but confirm before uploading.

**Per-slot `config.json` template** (mirrors slot B's, adapt values):
```json
{
  "slot": "<A|C|D|E>",
  "description": "<what variable this isolates, from §5>",
  "code_commit": "<git rev-parse HEAD short hash at train time>",
  "platform": {"provider": "Alibaba Cloud", "gpu": "NVIDIA L20 (48GB, Ada)", "torch": "...", "ultralytics_or_smp": "..."},
  "dataset": {"...": "same as slot B's, see §2"},
  "training_config": {"...": "exact FULL_*/UNET_*_CONFIG values used, see §5"},
  "final_training_metrics_val": {"...": "last-epoch mAP (YOLO) or dice/iou (U-Net)"},
  "test_set_pixel_metrics": {"...": "from results/seg_eval_*_test.json"}
}
```

**Upload commands** (from a machine with `hf auth login` write access —
**never inline the token in a command**, use the interactive login prompt):
```bash
hf upload johnnytan5/2nd-run-fyp-train <local_best.pt> <repo_path>/best_<slot>.pt --repo-type model
hf upload johnnytan5/2nd-run-fyp-train runs/<project>/exp <repo_path>/run --repo-type model
hf upload johnnytan5/2nd-run-fyp-train results <repo_path>/eval --repo-type model
```

---

## 9. Infrastructure history (context for the "why" behind some decisions)

- **Slot B** trained on **RunPod**, single **NVIDIA L4** (24GB), ~7min/epoch
  at 640px/batch16. Completed cleanly, results already on HF.
- **Slots A/C/D/E** trained on **Alibaba Cloud**, region hopped
  Singapore → Japan mid-session:
  - Singapore: CPU prep instance (Yitian/ARM) did data download + convert +
    split; GPU inventory (4×L20) sold out repeatedly down to 1×L20 by the
    time ready to train.
  - Migrated via **cross-region snapshot copy** (Singapore disk → snapshot →
    copied to Japan → new disk from snapshot) to reach guaranteed 4×L20
    availability in Japan (`ecs.gn8is-4x.16xlarge`, $13.27/hr).
  - **Near-miss**: the Japan disk was initially mistaken for blank and
    `mkfs.ext4`'d, which would have destroyed the migrated data — caught
    because the format warning showed a prior mount timestamp matching the
    Singapore disk. Recovered by creating a fresh disk from the still-intact
    snapshot (snapshots are independent of the disk's later state) and
    re-verifying the full split (27290/4376/5683, 0 broken symlinks) before
    resuming.
  - ARM-built `.venv` from the Singapore CPU box is **not reusable** on the
    x86 GPU instance — rebuilt fresh each time a new instance type is used.
  - `ultralytics` datasets_dir setting and Python venv must be **re-set on
    every fresh instance** (not preserved across instance swaps, even on the
    same persistent disk, since it's a per-OS-image config, not deployed
    with the disk itself unless explicitly baked in).

---

## 10. Open items

- [ ] Decide HF repo structure for A/C/D/E (§8) before first upload.
- [ ] `eval_segmentation.py` originally had no U-Net support — resolved via
      new `eval_unet.py` (reuses the same `Accum` metric math), but if any
      future eval needs to compare YOLO + U-Net predictions on the *same*
      image side by side, that's not built yet.
- [ ] No test-set mAP computed anywhere (only val, training-time) — pixel
      metrics are the only test-set numbers. Intentional (§4), but flag if a
      reviewer specifically asks for test-set mAP.
- [ ] FYP2 candidate levers not yet started: SAHI-tiled inference count via
      connected-component analysis, orthorectification for crack-panorama
      measurement, camera calibration (Mini 2 / Mini 4 Pro — likely already
      DJI-corrected, verify with a straight-edge test shot before assuming).
