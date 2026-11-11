# Case study 3: Pathology foundation models at the edge

**Question:** pathology foundation models are getting bigger (ViT-L, ViT-H), and bigger
backbones generalize better to new hospitals. But can a hospital or lab without a GPU server
actually run them? And if we compress them for a low-cost edge device, does the
out-of-distribution accuracy survive?

This benchmark runs six tile-level backbones on an NVIDIA Jetson in four formats and
measures **speed, power, energy, minutes per whole-slide image, embedding fidelity and
in-/out-of-distribution AUC** on CAMELYON17-WILDS.

| Backbone | Architecture | Params | Input | Access |
|---|---|---|---|---|
| ResNet-18 | CNN, ImageNet | 11 M | 224 | open |
| ViT-Tiny (Phikon-distil, Kather2016) | ViT-T/16 | 5.5 M | 224 | open |
| Phikon | ViT-B/16, TCGA SSL | 86 M | 224 | open |
| CONCH | ViT-B/16, vision-language | ~90 M | 448 | gated |
| UNI | ViT-L/16, 100M tiles SSL | 304 M | 224 | gated |
| Virchow | ViT-H/14, 1.5M WSI SSL | 632 M | 224 | gated |

Formats: **PyTorch FP32** (reference), **PyTorch FP16**, **TensorRT FP16**, **TensorRT INT8**
(entropy-calibrated on training tiles).

## How accuracy is measured
1. Sample balanced tumor/normal tiles from three CAMELYON17-WILDS splits: `id_train`,
   `id_val` (hospitals seen in training) and `ood_test` (an unseen hospital).
2. Extract FP32 embeddings and train a logistic-regression **linear probe** on `id_train`.
3. Apply that **same probe, unchanged**, to the embeddings from every format. This mirrors
   real deployment: you train on a workstation in FP32, then ship a compressed backbone to
   the edge device.
4. Report AUC on `id_val` and `ood_test`, the OOD AUC change vs FP32 with a paired bootstrap
   95% CI (1,000 resamples), and the mean cosine similarity of embeddings vs FP32.

## Setup (one time)
1. Use the same container as the other case studies (see the main README). Add
   `-e HF_TOKEN` to the `docker run` command if you will run gated models.
2. **Gated models:** log in to huggingface.co and accept the terms on each model page:
   MahmoodLab/UNI, MahmoodLab/conch and paige-ai/Virchow. Then create a read token
   (Settings → Access Tokens) and run `export HF_TOKEN=hf_...` before starting Docker.
3. **Data:** the script downloads one parquet shard per split (a few hundred MB) from the
   same Hugging Face mirror the manuscript uses. If you already downloaded CAMELYON17
   with the manuscript's `scripts/download_data.py`, copy or mount that folder and pass
   `--data-dir`.

## Run
```bash
bash pathology/run_pathology.sh                      # open models, about 20-30 min
HF_TOKEN=$HF_TOKEN bash pathology/run_pathology.sh   # all six; first run 1-3 h (engine builds)
```
TensorRT engines are cached in `pathology/engines/`, so later runs are much faster.
Useful options (`python3 pathology/run_pathology.py --help`):
- `--backbones uni conch` to run a subset
- `--n-train 1000 --n-test 1000` for tighter confidence intervals (slower)
- `--formats torch_fp32 trt_fp16` to skip formats
- `--board "Jetson Orin Nano Super"` to label results

The same script also runs on a laptop or desktop GPU, with or without TensorRT. That gives
a workstation-vs-edge comparison.

## Outputs (`pathology/results/`)
| File | What it is |
|---|---|
| `pathology_results.csv` | one row per backbone × format, all metrics |
| `pathology_report.md` | publication-style results table |
| `speed_vs_ood_auc.png` | deployability chart: throughput vs OOD AUC |

## Notes and limitations
- **Memory:** an 8 GB Orin Nano shares memory between CPU and GPU. Virchow (ViT-H, 632 M
  parameters) may not fit for FP32 export. If it is skipped, run it on a 16 GB+ Jetson
  (e.g. AGX Orin) or limit it to `--formats torch_fp16 trt_fp16`.
- **INT8 on ViTs:** TensorRT's post-training INT8 calibration keeps some transformer
  layers in FP16. Expect smaller speed-ups than for CNNs, and check the OOD AUC change.
- **Sample size:** the defaults (400 tiles per split) keep the run short; CIs reflect that.
- **Scope:** a linear probe on tile embeddings, not the full multi-agent models of the
  manuscript. It isolates the effect of the deployment format on the representation.
- **Throughput** is model time only. Reading and decoding slide tiles is extra.
- The CAMELYON17-WILDS copy is a third-party Hugging Face mirror; verify it against the
  official WILDS release before publication.
