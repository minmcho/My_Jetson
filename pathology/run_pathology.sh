#!/usr/bin/env bash
# Pathology edge benchmark. Run inside the Jetson container from the repo root:
#   bash pathology/run_pathology.sh                 # open models (ResNet-18, ViT-Tiny, Phikon)
#   HF_TOKEN=hf_xxx bash pathology/run_pathology.sh # + gated models (CONCH, UNI, Virchow)
# Extra arguments are passed through, e.g.  --data-dir /work/data/camelyon17
set -euo pipefail

python3 - <<'PY' || pip install timm transformers scikit-learn pyarrow pandas huggingface_hub
import timm, transformers, sklearn, pyarrow, pandas, huggingface_hub  # noqa
PY

BACKBONES="resnet18 vit_tiny phikon"
if [ -n "${HF_TOKEN:-}" ]; then
  python3 -c "import conch" 2>/dev/null || \
    pip install --no-deps "git+https://github.com/Mahmoodlab/CONCH.git" ftfy regex
  BACKBONES="$BACKBONES conch uni virchow"
fi

python3 pathology/run_pathology.py --backbones $BACKBONES "$@"
python3 pathology/make_pathology_report.py
