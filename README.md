## image object quantity metric (IOQM)

This repo contains code for:

1. building a dataset of image prompts with quantities of objects for evaluation
   - the objects are a set of 79/80 objects used in the COCO dataset. https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml

   - a mini version of the dataset containing ~16k images from a random subset of 10 COCO object classes lives here: https://huggingface.co/datasets/semaj83/ioqm
2. running various generative models through huggingface diffusers/transformers libraries
3. evaluating images generated using the original object detection model developed by facebook for the dataset: `facebook/detr-resnet-50`


### paper: TBD

### results: TBD

***code to create prompts***:

***code to generate images***:

***code to evaluate generated images***:

---

## v3 harness (2026 revival — see `docs/plan.md`)

The plan and literature review are in `docs/`. Phase 0 code:

| Path | What it is |
|---|---|
| `ioqm/classes.py` | Canonical COCO-80 table: hand-written plurals, aliases, and a flag for classes whose counting unit is ambiguous. Replaces prefix matching |
| `ioqm/prompts.py` | `PromptSpec` (ground truth stored as data, not in filenames), rendering, legacy prompt parser, nested v3 builder, stratified evaluation subset. Splits: `train`, `test_id`, `test_single`, `test_ood_pair` (a class pair never seen together in any training prompt), `test_ood_triple` (a new triple made of familiar pairs), `test_ood_count` |
| `ioqm/metrics.py` | Fixed metrics (primary: `prompt_exact`) and cluster/paired bootstrap CIs |
| `ioqm/generate.py` | Model registry and generation that resumes after a disconnect (`manifest.jsonl` + `run_config.json`) |
| `ioqm/detect.py` | Stores raw detections at a low confidence floor, then counts at any global or per-class threshold |
| `scripts/` | `build_prompts_v3.py`, `generate.py`, `benchmark_speed.py`, `import_legacy_run.py`, `score.py` |

`run_eval.py`'s `hard_ioqm` / `soft_ioqm` are kept **unchanged** so the 2023 numbers can be reproduced. Their bugs are documented in the docstring.

```bash
python -m pytest tests -q                 # no GPU needed
python scripts/build_prompts_v3.py        # -> prompts/v3/{all,train,eval_subset}.jsonl
```

### Colab (A100)

```python
!nvidia-smi                                   # note whether the A100 is 40GB or 80GB
!git clone -b phase0-harness https://github.com/semajyllek/ioqm.git   # only works once the branch is pushed
%cd ioqm
!pip install -r requirements.txt              # detectors: ultralytics==8.0.28 / sahi pins are from 2023 and untested with current torch
!pip install -U diffusers transformers accelerate sentencepiece protobuf   # new pipelines (Flux2Klein, QwenImage, ZImage) need recent diffusers
from huggingface_hub import login; login()    # needed for gated models: SD3.5, FLUX.1-dev, FLUX.2-klein-9B (accept each license on HF first)
from google.colab import drive; drive.mount('/content/drive')

# 1) time every registered model (results are appended per model)
!python scripts/benchmark_speed.py --out /content/drive/MyDrive/ioqm_data/speed.jsonl

# 2) generate (re-run the same command after a disconnect to resume)
!python scripts/generate.py --model z_image_turbo --prompts prompts/v3/eval_subset.jsonl \
    --seeds 0 1 2 3 --out_dir /content/drive/MyDrive/ioqm_data/runs/v3_eval/z_image_turbo

# 3) re-score a 2023 run with the fixed metrics
!python scripts/import_legacy_run.py --model sd15 --dataset v2 \
    --img_folder /content/drive/MyDrive/ioqm_data/generated_images_v2/stable-diffusion-v1-5_images \
    --out_dir /content/drive/MyDrive/ioqm_data/runs/legacy_v2/sd15
!python scripts/score.py --run_dir /content/drive/MyDrive/ioqm_data/runs/legacy_v2/sd15 --detector yolo
```
