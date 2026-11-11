"""CAMELYON17-WILDS tiles for the benchmark (same source as the manuscript).

Splits used:
  id_train  -> fit the linear probe and calibrate INT8 engines
  id_val    -> in-distribution test (hospitals seen in training)
  ood_test  -> out-of-distribution test (unseen hospital)

Either point --data-dir at the folder your manuscript pipeline already downloaded
(data/camelyon17 from scripts/download_data.py), or let this module download ONE
parquet shard per split from Hugging Face (a few hundred MB instead of the full set).
Tiles are cached as uint8 arrays, resized to the backbone input size.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image

HF_REPO = "wltjr1007/Camelyon17-WILDS"
SPLITS = ["id_train", "id_val", "ood_test"]


def _split_files(split, data_dir, cache_dir):
    if data_dir:
        files = sorted(p for p in Path(data_dir).rglob("*.parquet") if split in p.name or split in str(p.parent.name))
        if not files:
            raise SystemExit(f"No parquet files for split '{split}' under {data_dir}")
        return files
    from huggingface_hub import HfApi, hf_hub_download
    names = sorted(f for f in HfApi().list_repo_files(HF_REPO, repo_type="dataset")
                   if f.endswith(".parquet") and split in f)
    if not names:
        raise SystemExit(f"Could not find split '{split}' in {HF_REPO}")
    print(f"[data] downloading 1 of {len(names)} shard(s) for {split}: {names[0]}")
    return [Path(hf_hub_download(HF_REPO, names[0], repo_type="dataset", local_dir=cache_dir))]


def _sample(files, n, seed):
    """Balanced sample of n tiles (n/2 per class) from the first files that have enough."""
    import pyarrow.parquet as pq
    import pandas as pd
    frames, have = [], {0: 0, 1: 0}
    for f in files:
        df = pq.read_table(f, columns=["image", "label"]).to_pandas()
        frames.append(df)
        for k in have:
            have[k] += int((df["label"] == k).sum())
        if min(have.values()) >= n // 2:
            break
    df = pd.concat(frames, ignore_index=True)
    rng = np.random.RandomState(seed)
    idx = []
    for k in (0, 1):
        pool = df.index[df["label"] == k].to_numpy()
        idx.append(rng.choice(pool, min(n // 2, len(pool)), replace=False))
    sel = np.concatenate(idx)
    rng.shuffle(sel)
    return df.iloc[sel].reset_index(drop=True)


def _decode(cell, size):
    data = cell.get("bytes") if isinstance(cell, dict) else cell
    img = Image.open(io.BytesIO(data)).convert("RGB")
    # Same as the manuscript pipeline: Resize(size) + CenterCrop(size)
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.uint8)


def load_tiles(split, n, size, data_dir=None, cache_dir="pathology/data_cache", seed=42):
    """Return (images uint8 [N, S, S, 3], labels int64 [N])."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{split}_n{n}_s{size}_seed{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["images"], z["labels"]
    df = _sample(_split_files(split, data_dir, cache_dir / "raw"), n, seed)
    images = np.stack([_decode(c, size) for c in df["image"]])
    labels = df["label"].to_numpy().astype(np.int64)
    np.savez(cache, images=images, labels=labels)
    print(f"[data] {split}: {len(labels)} tiles at {size}px "
          f"({int(labels.sum())} tumor / {int((labels == 0).sum())} normal)")
    return images, labels
