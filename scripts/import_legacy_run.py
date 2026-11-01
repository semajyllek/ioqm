"""
Wrap a 2023 image folder (filenames like 3_broccolis_and_1_toaster.png) as a v3
run directory, so the old images can be re-scored with the fixed metrics and
canonical class matching. The images are symlinked (or referenced through images_path.txt where
symlinks are unsupported), never copied or moved.

    python scripts/import_legacy_run.py \
        --img_folder /content/drive/MyDrive/ioqm_data/generated_images_v2/stable-diffusion-v1-5_images \
        --model sd15 --dataset v2 \
        --out_dir /content/drive/MyDrive/ioqm_data/runs/legacy_v2/sd15

Seed is recorded as -1 (unknown): the 2023 runs did not fix seeds.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.prompts import make_spec, parse_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_folder", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["v1", "v2"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    src, out = Path(args.img_folder).resolve(), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    link = out / "images"
    if not link.exists() and not (out / "images_path.txt").exists():
        try:
            os.symlink(src, link, target_is_directory=True)
        except OSError:  # e.g. Colab's Drive mount has no symlink support
            (out / "images_path.txt").write_text(str(src))

    n_ok, bad = 0, []
    with open(out / "manifest.jsonl", "w") as f:
        for p in sorted(src.glob("*.png")):
            try:
                objs = list(parse_prompt(p.stem).items())
            except ValueError as e:
                bad.append((p.name, str(e)))
                continue
            spec = make_spec(objs, dataset=args.dataset, split="legacy")
            f.write(json.dumps({
                "image": p.name, "prompt_id": spec.prompt_id, "seed": -1,
                "prompt": p.stem.replace("_", " "), "objects": [list(o) for o in spec.objects],
                "split": "legacy", "dataset": args.dataset, "model": args.model,
            }) + "\n")
            n_ok += 1
    print(f"imported {n_ok} images; {len(bad)} unparseable")
    for name, err in bad[:20]:
        print("  ", name, err)


if __name__ == "__main__":
    main()
