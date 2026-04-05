#!/usr/bin/env python3
"""Download all 24 Jackrong datasets from HuggingFace into High-fidelity Dataset folder."""

import os
from huggingface_hub import snapshot_download, HfApi

TARGET_DIR = os.path.join(os.path.dirname(__file__), "High-fidelity Dataset")
os.makedirs(TARGET_DIR, exist_ok=True)

# Get all datasets from Jackrong
api = HfApi()
datasets = list(api.list_datasets(author="Jackrong"))
print(f"Found {len(datasets)} datasets from Jackrong\n")

for i, ds in enumerate(datasets, 1):
    ds_name = ds.id.split("/")[-1]  # e.g. "Qwen3.5-reasoning-700x"
    local_dir = os.path.join(TARGET_DIR, ds_name)
    
    print(f"[{i}/{len(datasets)}] Downloading: {ds.id}")
    print(f"  -> {local_dir}")
    
    try:
        snapshot_download(
            repo_id=ds.id,
            repo_type="dataset",
            local_dir=local_dir,
        )
        print(f"  ✓ Done\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print("=" * 60)
print(f"All downloads complete! Files are in: {TARGET_DIR}")
