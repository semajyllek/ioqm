"""Generates notebooks/phase0_colab.ipynb:  python tools/make_phase0_notebook.py notebooks/phase0_colab.ipynb"""
import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": s.strip("\n").splitlines(keepends=True)})

md("""
# IOQM: Colab runner

Run it top to bottom: **Runtime → Run all**. Runtime type: **A100 GPU**.

Every step can be resumed. Results live on Google Drive under `MyDrive/ioqm_data/`. After a disconnect, run it top to bottom again: finished images, detections and timings are skipped, and work picks up where it stopped.

If a step fails, the notebook stops there with an error instead of carrying on.

**Before starting:** if the old notebook is still generating in another Colab session, stop that session first (Runtime → Disconnect and delete runtime). Two sessions writing to the same run folder would duplicate work.

| Section | What it does | GPU time |
|---|---|---|
| 1 | Time any model not yet timed (only the four gated ones are left) | ~1 h incl. downloads |
| 2 | **Best-of-N pilot** with FLUX.2-klein-4B: 480 prompts × 16 seeds | ~3–4 h |
| 3 | Finish the z_image_turbo eval run (2,985 prompts × 4 seeds), score it, best-of-4 | whatever is left of ~13 h |
| 4 | Re-score the 2023 SD1.5 v2 images with the fixed metrics | ~15 min |
| 5 | **Phase 1:** counters vs your 2023 human labels; build the labelling queue; run the VLM counter on the pilot | ~2–3 h |

Sections 1–4 have already run, so they are switched off; switching one back on just skips finished work. Each section has an on/off switch in the settings cell below.

Labelling itself happens in `notebooks/label_colab.ipynb`, which needs no GPU.
""")

md("## 0. Settings and setup")

code("""
# ---- Settings ---------------------------------------------------------------
RUN_TIMING   = False   # section 1 (done)
RUN_BON      = False   # section 2: best-of-N pilot (done)
RUN_EVAL     = False   # section 3: z_image_turbo eval run (done)
RUN_LEGACY   = False   # section 4: 2023 re-score (done)
RUN_PHASE1   = True    # section 5: counters vs humans, labelling queue, VLM counter
QUEUE_SIZE   = 300
VLM_COUNTER  = 'qwen3_vl_8b'

BON_MODEL    = 'flux2_klein_4b'   # fastest model in the timing table (1.08 s/img)
BON_SEEDS    = 16                 # 8 halves the time
EVAL_MODEL   = 'z_image_turbo'    # matches the run already started on Drive
EVAL_SEEDS   = [0, 1, 2, 3]
COUNTERS     = ['yolov8x', 'mask2former']
LEGACY_FOLDER = 'generated_images_v2/stable-diffusion-v1-5_images'   # relative to DATA

REPO_URL, BRANCH = 'https://github.com/semajyllek/ioqm.git', 'phase0-harness'
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv

from google.colab import drive
drive.mount('/content/drive')
import os
DATA = '/content/drive/MyDrive/ioqm_data'
os.makedirs(DATA, exist_ok=True)
""")

code("""
# Fresh copy of the code every time, so a stale checkout can never be used.
%cd /content
!rm -rf /content/ioqm && git clone -q -b $BRANCH $REPO_URL /content/ioqm
%cd /content/ioqm
!git log --oneline -1
""")

code("""
# Not quiet on purpose: if an install fails you should see it.
!pip install -U diffusers transformers accelerate sentencepiece protobuf 2>&1 | tail -n 3
!pip install -r requirements.txt pytest 2>&1 | tail -n 3

import importlib
for mod in ('ultralytics', 'diffusers', 'transformers'):
    print(f'{mod:13s}', importlib.import_module(mod).__version__)
""")

code("""
# Helper: run a shell command, stream its output, and STOP the notebook if it fails.
import subprocess, sys
def sh(cmd):
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    for line in p.stdout:
        print(line, end='', flush=True)
    if p.wait() != 0:
        raise RuntimeError(f'command failed (exit {p.returncode}): {cmd}')

sh('python -m pytest tests -q')
""")

md("""
### Hugging Face login

Uses the Colab secret `HF_TOKEN` if you have one (key icon in the left sidebar, with notebook access switched on). Otherwise it asks you to paste a token. A plain **Read** token works.

The token is also put in the environment, so the scripts this notebook launches can use it. The four 401 errors earlier were the timing loop skipping models it had already tried, not the token, but this rules the token out as well.
""")

code("""
from huggingface_hub import login, whoami
try:
    from google.colab import userdata
    token = userdata.get('HF_TOKEN')
except Exception:
    token = None
if token:
    login(token=token)
else:
    login()                       # paste the token when prompted
    from huggingface_hub import get_token
    token = get_token()
os.environ['HF_TOKEN'] = token
print('Logged in as:', whoami()['name'])
""")

code("""
# Which gated models can this account download? Models without access are left out of section 1.
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from ioqm.generate import REGISTRY
GATED = ['sd35_medium', 'sd35_large', 'flux1_dev', 'flux2_klein_9b']
blocked = []
for m in GATED:
    repo = REGISTRY[m].hf_id
    try:
        hf_hub_download(repo, 'model_index.json')
        print(f'OK       {m:15s} {repo}')
    except (GatedRepoError, RepositoryNotFoundError) as e:
        blocked.append(m)
        print(f'NO ACCESS {m:15s} {repo}  -> accept the license at https://huggingface.co/{repo}  ({type(e).__name__})')
    except Exception as e:   # e.g. the file name differs; access itself is not the problem
        print(f'?        {m:15s} {repo}  could not check ({type(e).__name__}); timing will try anyway')
""")

md("""
## 1. Time every model not yet timed

Loads each model, makes 1 warm-up and 5 timed images, and appends seconds per image, peak memory and offload mode to `DATA/speed.jsonl`. Models that already have a successful timing are skipped; failed ones are retried. Each model's weights are deleted from local disk after timing.
""")

code("""
import json, shutil
SPEED = f'{DATA}/speed.jsonl'
MODELS = ['sd15', 'sdxl_base', 'z_image_turbo', 'flux2_klein_4b', 'sd35_medium', 'z_image',
          'flux2_klein_9b', 'sd35_large', 'flux1_dev', 'qwen_image', 'sdxl_base_refiner']

def speed_rows():
    return [json.loads(l) for l in open(SPEED) if l.strip()] if os.path.exists(SPEED) else []

if RUN_TIMING:
    done = {r['model'] for r in speed_rows() if r.get('status') == 'ok'}
    HF_CACHE = os.path.expanduser('~/.cache/huggingface/hub')
    for m in MODELS:
        if m in done:
            print(f'skip {m} (already timed)'); continue
        if m in blocked:
            print(f'skip {m} (no Hugging Face access yet)'); continue
        try:
            sh(f'python scripts/benchmark_speed.py --models {m} --out {SPEED}')
        except RuntimeError as e:
            print('timing failed, continuing:', e)   # recorded in speed.jsonl; retried next run
        if m not in (BON_MODEL, EVAL_MODEL):          # keep the weights we are about to use
            shutil.rmtree(os.path.join(HF_CACHE, 'models--' + REGISTRY[m].hf_id.replace('/', '--')), ignore_errors=True)
""")

code("""
import pandas as pd
df = pd.DataFrame(speed_rows()).drop_duplicates('model', keep='last')   # latest attempt per model
cols = [c for c in ['model', 'status', 'offload', 's_per_img_median', 'peak_mem_gb',
                    'projected_gpu_hours_12k', 'load_s', 'error'] if c in df]
df[cols].sort_values('s_per_img_median', na_position='last')
""")

md("""
## 2. Best-of-N pilot

1. Generate the 480 pilot prompts × 16 seeds with `BON_MODEL`.
2. Detect every image with both counters. All detections with score ≥ 0.05 are stored, so thresholds can change later without re-running anything.
3. For k = 1, 2, 4, 8, 16, one counter (the **verifier**) picks the best of k images and the **other** counter (the **evaluator**) scores the pick. Both role assignments are reported.
""")

code("""
BON_RUN = f'{DATA}/runs/bon_pilot/{BON_MODEL}'
EVAL_RUN = f'{DATA}/runs/v3_eval/{EVAL_MODEL}'
if RUN_BON:
    seeds = ' '.join(map(str, range(BON_SEEDS)))
    sh(f'python scripts/generate.py --model {BON_MODEL} --prompts prompts/v3/pilot.jsonl --seeds {seeds} --out_dir {BON_RUN}')
    for c in COUNTERS:
        sh(f'python scripts/score.py --run_dir {BON_RUN} --detector {c}')
    ks = ' '.join(str(k) for k in (1, 2, 4, 8, 16) if k <= BON_SEEDS)
    sh(f'python scripts/best_of_n.py --run_dir {BON_RUN} --counters {" ".join(COUNTERS)} --ks {ks} --by n_types')
""")

md("""
**How to read it.**
- **`selected (eval)`** is the result: count accuracy of the verifier's pick, judged by the other counter.
- **`gain vs k=1`** is the improvement over a single sample, with a paired 95% CI.
- **`verifier self`** is how good the verifier *thinks* its pick is. A big gap above `selected` means the verifier is being gamed.
- **`oracle/pass@k`** is the ceiling a perfect verifier would reach.
- **`[SELF]`** blocks, where verifier and evaluator are the same counter, are not results.

If the two role assignments disagree a lot, the counters disagree on these images. Checking against the human labels decides which to trust.
""")

md("""
## 3. z_image_turbo evaluation run

Resumes the run already on Drive (2,985 prompts × 4 seeds), scores it with both counters, and computes best-of-1/2/4 from the 4 seeds.
""")

code("""
EVAL_RUN = f'{DATA}/runs/v3_eval/{EVAL_MODEL}'
if RUN_EVAL:
    seeds = ' '.join(map(str, EVAL_SEEDS))
    sh(f'python scripts/generate.py --model {EVAL_MODEL} --prompts prompts/v3/eval_subset.jsonl --seeds {seeds} --out_dir {EVAL_RUN}')
    for c in COUNTERS:
        sh(f'python scripts/score.py --run_dir {EVAL_RUN} --detector {c}')
    ks = ' '.join(str(k) for k in (1, 2, 4) if k <= len(EVAL_SEEDS))
    sh(f'python scripts/best_of_n.py --run_dir {EVAL_RUN} --counters {" ".join(COUNTERS)} --ks {ks} --by split')
""")

md("""
## 4. Re-score the 2023 SD1.5 v2 images

Wraps the old image folder as a run (nothing is copied), detects with YOLOv8x, and prints the fixed metrics. Then re-scores at two other thresholds without re-running the detector.
""")

code("""
LEGACY_RUN = f'{DATA}/runs/legacy_v2/sd15'
if RUN_LEGACY:
    src = f'{DATA}/{LEGACY_FOLDER}'
    if not os.path.isdir(src):
        print('legacy folder not found, skipping:', src)
    else:
        sh(f'python scripts/import_legacy_run.py --img_folder "{src}" --model sd15 --dataset v2 --out_dir "{LEGACY_RUN}"')
        sh(f'python scripts/score.py --run_dir "{LEGACY_RUN}" --detector yolov8x')
        for t in (0.3, 0.7):
            sh(f'python scripts/score.py --run_dir "{LEGACY_RUN}" --detector yolov8x --skip_detect --threshold {t}')
""")

md("""
## 5. Phase 1: how good are the counters?

**5a.** Score YOLOv8x, Mask2Former and the VLM counter (Qwen3-VL-8B) against your 2023 human labels in `eval/human_labels/`. No new labelling. Also prints a threshold sweep and, for images you labelled twice, your own consistency.

**5b.** Build the labelling queue: 300 images from the klein-4B pilot and the z_image_turbo run, oversampling images where YOLO and Mask2Former disagree, with agreement controls. Written to `DATA/labels/queue_v1.jsonl`. Then open `notebooks/label_colab.ipynb` (a CPU runtime is enough) and label.

**5c.** Run the VLM counter on the whole pilot run (about 2 h) plus the queued z_image_turbo images, and compare all three counters with each other.
""")

code("""
LABEL_DIR = f'{DATA}/labels'
QUEUE = f'{LABEL_DIR}/queue_v1.jsonl'
if RUN_PHASE1:
    # 5a: counters vs the 2023 human labels
    sh(f'python scripts/human_check.py --data {DATA} --counters yolov8x mask2former {VLM_COUNTER}')
""")

code("""
if RUN_PHASE1:
    # 5b: labelling queue (same arguments -> same queue; never overwritten once it exists)
    os.makedirs(LABEL_DIR, exist_ok=True)
    if os.path.exists(QUEUE):
        print('queue already exists, keeping it:', QUEUE)
    else:
        sh(f'python scripts/sample_for_labeling.py --runs {BON_RUN} {EVAL_RUN} '
           f'--counters yolov8x mask2former --n {QUEUE_SIZE} --out {QUEUE}')
""")

code("""
if RUN_PHASE1:
    # 5c: third counter on the pilot run, then three-way counter agreement
    sh(f'python scripts/score.py --run_dir {BON_RUN} --detector {VLM_COUNTER}')
    # on the eval run, only the queued images (minutes instead of ~4 h)
    sh(f'python scripts/score.py --run_dir {EVAL_RUN} --detector {VLM_COUNTER} --from_queue {QUEUE}')
    sh(f'python scripts/agreement_report.py --runs {BON_RUN} --counters yolov8x mask2former {VLM_COUNTER}')
""")

md("""
## 6. After labelling: counters vs your labels

Run this once you have labelled a good share of the queue (it works on a partial set). It reports each counter's agreement with you, both on the sample as drawn and re-weighted to the whole run.
""")

code("""
LABELS = f'{LABEL_DIR}/labels.jsonl'
if os.path.exists(LABELS):
    sh(f'python scripts/agreement_report.py --labels {LABELS} --queue {QUEUE} '
       f'--counters yolov8x mask2former {VLM_COUNTER}')
else:
    print('no labels yet:', LABELS)
""")

md("""
## Done

Send back the outputs of sections 2 and 3, and the timing table if new models were timed. Summaries are also saved to Drive:
- `runs/bon_pilot/<model>/best_of_n/summary.json`
- `runs/v3_eval/<model>/best_of_n/summary.json`
- `runs/*/scores/*.jsonl` (per-image scores)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
open(sys.argv[1], "a").write("\n")
