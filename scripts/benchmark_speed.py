"""
Time each registered model on this GPU. Run it once in Colab; the results
decide the Phase-2 model list and seed count (docs/plan.md, Decisions §2).

    python scripts/benchmark_speed.py --out speed.jsonl            # all models
    python scripts/benchmark_speed.py --models sd15 z_image_turbo  # some models

For each model it records the load time, offload mode used, seconds per image
(after 1 warm-up image), peak GPU memory, and the projected GPU-hours for
12k images (the ~3k-prompt x 4-seed evaluation subset). On out-of-memory it
retries with model CPU offload, and after that with sequential offload.
Results are appended one model at a time, so a disconnect loses at most the
model being timed.
"""

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.generate import REGISTRY, _versions, load_backend
from ioqm.prompts import make_spec, render

PROMPTS = [
    make_spec([("dog", 3)]),
    make_spec([("cup", 2), ("apple", 4)]),
    make_spec([("person", 1), ("umbrella", 3), ("teddy bear", 2)]),
    make_spec([("car", 5), ("elephant", 2)]),
    make_spec([("bottle", 4)]),
]
N_EVAL_IMAGES = 12_000


def time_model(key: str, n_images: int, offload: str) -> dict:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    backend = load_backend(REGISTRY[key], offload=offload)
    load_s = time.time() - t0
    try:
        backend.generate(render(PROMPTS[0]), seed=123)  # warm-up
        times = []
        for i in range(n_images):
            torch.cuda.synchronize()
            t = time.time()
            backend.generate(render(PROMPTS[i % len(PROMPTS)]), seed=i)
            torch.cuda.synchronize()
            times.append(time.time() - t)
        s_per_img = sorted(times)[len(times) // 2]
        return {
            "model": key, "status": "ok", "offload": getattr(backend, "offload", offload),
            "load_s": round(load_s, 1), "s_per_img_median": round(s_per_img, 2),
            "s_per_img_all": [round(x, 2) for x in times],
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "projected_gpu_hours_12k": round(s_per_img * N_EVAL_IMAGES / 3600, 1),
        }
    finally:
        del backend
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=sorted(REGISTRY))
    ap.add_argument("--n_images", type=int, default=5)
    ap.add_argument("--out", default="speed.jsonl")
    args = ap.parse_args()

    import torch

    print(json.dumps(_versions(), indent=2))
    for key in args.models:
        result = None
        for offload in ("auto", "model", "sequential"):
            try:
                print(f"\n=== {key} (offload={offload})")
                result = time_model(key, args.n_images, offload)
                break
            except torch.cuda.OutOfMemoryError:
                print(f"OOM with offload={offload}")
                gc.collect()
                torch.cuda.empty_cache()
                result = {"model": key, "status": "oom", "offload_tried": offload}
            except Exception as e:  # gated repo, missing pipeline class, etc.
                traceback.print_exc()
                result = {"model": key, "status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
                break
        result["versions"] = _versions()
        print(json.dumps({k: v for k, v in result.items() if k != "versions"}))
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
