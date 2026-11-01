"""
Generate images for a prompt JSONL with one registered model. Resumable: if
Colab disconnects, re-run the same command.

    python scripts/generate.py --model z_image_turbo \
        --prompts prompts/v3/eval_subset.jsonl --seeds 0 1 2 3 \
        --out_dir /content/drive/MyDrive/ioqm_data/runs/v3_eval/z_image_turbo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.generate import REGISTRY, load_backend, run_generation
from ioqm.prompts import load_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--template", default="{objects}")
    ap.add_argument("--joiner", default=" and ")
    ap.add_argument("--numerals", default="digits", choices=["digits", "words"])
    ap.add_argument("--offload", default="auto", choices=["auto", "none", "model", "sequential"])
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--split", default=None, help="only prompts in this split")
    args = ap.parse_args()

    specs = load_specs(Path(args.prompts))
    if args.split:
        specs = [s for s in specs if s.split == args.split]
    backend = load_backend(REGISTRY[args.model], offload=args.offload)
    run_generation(backend, specs, args.seeds, Path(args.out_dir), template=args.template,
                   joiner=args.joiner, numerals=args.numerals, max_images=args.max_images)


if __name__ == "__main__":
    main()
