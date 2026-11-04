"""
Generates the Colab notebooks from source, so notebook code is reviewable in diffs.

    python tools/make_notebooks.py          # rewrites notebooks/overopt_colab.ipynb

Edit this file, not the .ipynb.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook(cells, gpu=True):
    meta = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}}
    if gpu:
        meta.update({"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []}})
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


def write(nb, path):
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", path.relative_to(ROOT))


SETUP = [
    code("""
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs(DATA, exist_ok=True)

%cd /content
!rm -rf /content/ioqm && git clone -q -b $BRANCH https://github.com/semajyllek/ioqm.git /content/ioqm
%cd /content/ioqm
!git log --oneline -1
"""),
    code("""
# Not quiet on purpose: a failed install should be visible.
!pip install -U diffusers transformers accelerate sentencepiece protobuf 2>&1 | tail -n 2
!pip install -r requirements.txt pytest statsmodels matplotlib 2>&1 | tail -n 2
from ioqm.colab import sh, hf_login, gpu_memory_gb, dir_size_gb
sh('python -m pytest tests -q')
print('Hugging Face user:', hf_login())
"""),
]


OVEROPT = [
    md("""
# IOQM: does best-of-k against a counter improve counting, or only the counter's opinion?

Each **proxy** counter (YOLOv8x, Mask2Former, and an ensemble requiring both to agree) picks the best of k images per prompt, k = 1 … 64. An independent **gold** judge (a VLM that is never used for picking) scores the picks. If gold accuracy flattens while the proxy's own score keeps rising, optimizing against the counter is mostly fooling it.

The gold judge is only as trustworthy as its agreement with **you**. Two labelling rounds provide that evidence, and the paper rests on them:
1. `queue_v1` (from the main notebook): validates the gold judge against your labels.
2. `queue_picks` (step H here): your verdicts on the actual picks at k = 1 and k = 64.

**Run all** on an A100. Every step resumes, so after a disconnect, run all again. Generation alone is about 10 h, so expect several sessions. The curve cells work on whatever is finished.

| Step | What | GPU time |
|---|---|---|
| A | First curve from the 16 seeds that already exist | minutes |
| B | Drive space check | – |
| C | Generate seeds 16–63 for the 480 pilot prompts (23,040 images) | ~10 h |
| D | Proxy counters on the new images | ~1 h |
| E | Gold judge: queued images first, then the whole pilot | ~3–9 h (8B / 32B) |
| F | Gold judge vs your labels | minutes |
| G | Full curve, k up to 64 | minutes |
| H | Queue of real picks for you to label | – |
| I | Your verdicts vs gold vs proxy | – |
| J | Mechanism analyses (independence, trading, what predicts failure) | CPU, ~15 min |

Not in this notebook: RL fine-tuning. That needs a separate pilot first.
"""),
    md("## Settings"),
    code("""
DATA   = '/content/drive/MyDrive/ioqm_data'
BRANCH = 'phase0-harness'

MODEL    = 'flux2_klein_4b'
RUN      = f'{DATA}/runs/bon_pilot/{MODEL}'      # the 480-prompt pilot (seeds 0-15 already exist)
EVAL_RUN = f'{DATA}/runs/v3_eval/z_image_turbo'  # 2,985 prompts x 4 seeds (regression only)
N_SEEDS  = 64
PROXIES  = ['yolov8x', 'mask2former', 'ens:yolov8x+mask2former']
GOLD     = 'auto'     # 'auto' = qwen3_vl_32b on an 80 GB GPU, else qwen3_vl_8b. Fixed after the first run.

LABEL_DIR    = f'{DATA}/labels'
QUEUE_V1     = f'{LABEL_DIR}/queue_v1.jsonl'      # built by the main notebook, section 5
LABELS_V1    = f'{LABEL_DIR}/labels.jsonl'
QUEUE_PICKS  = f'{LABEL_DIR}/queue_picks.jsonl'
LABELS_PICKS = f'{LABEL_DIR}/labels_picks.jsonl'
PICKS_PROMPTS = 50

STEPS = dict(A=True, B=True, C=True, D=True, E=True, F=True, G=True, H=True, I=True, J=True)
"""),
    md("## Setup"),
    *SETUP,
    code("""
# Choose the gold judge once and remember it, so a later session on a different GPU
# cannot silently switch judges halfway through the experiment.
import json
SETTINGS = f'{RUN}/overopt/settings.json'
os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
mem = gpu_memory_gb()
print(f'GPU memory: {mem:.0f} GB')
if os.path.exists(SETTINGS):
    GOLD = json.load(open(SETTINGS))['gold']
    print('gold judge (fixed by an earlier session):', GOLD)
else:
    if GOLD == 'auto':
        GOLD = 'qwen3_vl_32b' if mem >= 75 else 'qwen3_vl_8b'
    json.dump({'gold': GOLD, 'proxies': PROXIES, 'n_seeds': N_SEEDS}, open(SETTINGS, 'w'))
    print('gold judge chosen and saved:', GOLD)
if GOLD == 'qwen3_vl_32b' and mem < 75:
    raise RuntimeError('this experiment uses the 32B gold judge, which needs an 80 GB A100; switch runtime')
"""),
    md("## A. First curve from the existing 16 seeds"),
    code("""
gold_file = f'{RUN}/scores/{GOLD}__t0.5.jsonl'
if STEPS['A']:
    if os.path.exists(gold_file):
        sh(f'python scripts/overopt_curve.py --run_dir {RUN} --gold {GOLD} --proxies {" ".join(PROXIES)} --ks 1 2 4 8 16')
    else:
        print(f'no {GOLD} scores yet; the curve appears after step E')
"""),
    md("## B. Drive space check"),
    code("""
if STEPS['B']:
    import shutil
    n_now = sum(1 for l in open(f'{RUN}/manifest.jsonl') if l.strip())
    size_now = dir_size_gb(f'{RUN}/images')
    need = size_now / max(n_now, 1) * (480 * N_SEEDS - n_now)
    free = shutil.disk_usage('/content/drive/MyDrive').free / 2**30
    print(f'{n_now} images use {size_now:.1f} GB; the remaining {480 * N_SEEDS - n_now} need about {need:.1f} GB')
    print(f'free space reported for the Drive mount: {free:.0f} GB (approximate; check your Google storage page)')
    if free < need * 1.2:
        print('WARNING: Drive may fill up before generation finishes')
"""),
    md("## C. Generate seeds up to 64 (resumes; ~10 h)"),
    code("""
if STEPS['C']:
    seeds = ' '.join(map(str, range(N_SEEDS)))
    sh(f'python scripts/generate.py --model {MODEL} --prompts prompts/v3/pilot.jsonl --seeds {seeds} --out_dir {RUN}')
"""),
    md("## D. Proxy counters on the new images"),
    code("""
if STEPS['D']:
    for c in ('yolov8x', 'mask2former'):
        sh(f'python scripts/score.py --run_dir {RUN} --detector {c}')
"""),
    md("""
## E. Gold judge

First on the images in your labelling queue (both runs), so it can be checked against your labels early; then on the whole pilot.
"""),
    code("""
if STEPS['E']:
    if os.path.exists(QUEUE_V1):
        for r in (RUN, EVAL_RUN):
            sh(f'python scripts/score.py --run_dir {r} --detector {GOLD} --from_queue {QUEUE_V1}')
    else:
        print('no queue_v1 yet (main notebook, section 5b); skipping the queued-images pass')
    sh(f'python scripts/score.py --run_dir {RUN} --detector {GOLD}')
"""),
    md("## F. Gold judge vs your labels"),
    code("""
if STEPS['F']:
    if os.path.exists(LABELS_V1):
        sh(f'python scripts/agreement_report.py --labels {LABELS_V1} --queue {QUEUE_V1} '
           f'--counters yolov8x mask2former {GOLD}')
    else:
        print('no labels yet: label queue_v1 in notebooks/label_colab.ipynb')
"""),
    md("## G. Full curve"),
    code("""
if STEPS['G']:
    sh(f'python scripts/overopt_curve.py --run_dir {RUN} --gold {GOLD} --proxies {" ".join(PROXIES)} --by n_types')
    from IPython.display import Image, display
    png = f'{RUN}/overopt/curves__gold-{GOLD}.png'
    if os.path.exists(png):
        display(Image(png))
"""),
    md("""
## H. Queue of real picks for you to label

For 50 random prompts, each proxy's actual pick at k = 1 and k = 64. Only built once every judge has scored all 64 seeds of enough prompts.

Label it in `notebooks/label_colab.ipynb` with **`QUEUE = queue_picks.jsonl` and `LABELS = labels_picks.jsonl`** (keep the two labelling rounds in separate files).
"""),
    code("""
if STEPS['H']:
    if os.path.exists(QUEUE_PICKS):
        print('picks queue already exists:', QUEUE_PICKS)
    else:
        sh(f'python scripts/picks_queue.py --run_dir {RUN} --gold {GOLD} --proxies {" ".join(PROXIES)} '
           f'--ks 1 {N_SEEDS} --n_prompts {PICKS_PROMPTS} --out {QUEUE_PICKS}')
"""),
    md("## I. Your verdicts on the picks"),
    code("""
if STEPS['I']:
    if os.path.exists(LABELS_PICKS):
        sh(f'python scripts/picks_report.py --queue {QUEUE_PICKS} --labels {LABELS_PICKS}')
    else:
        print('no picks labels yet')
"""),
    md("""
## J. Mechanism

- Are per-type errors independent? Do types trade instances? (pilot run: needs many seeds per prompt)
- What predicts failure: number of types, largest count, total instances, held-out pairs, how often the classes co-occur in COCO (both runs)
"""),
    code("""
COOC = f'{DATA}/coco_cooccurrence.json'
if STEPS['J']:
    if not os.path.exists(COOC):
        sh(f'python scripts/coco_cooccurrence.py --out {COOC}')
    from ioqm.counters import DEFAULT_THRESHOLDS
    judges = ' '.join(j for j in (GOLD, 'yolov8x', 'mask2former')
                      if os.path.exists(f'{RUN}/scores/{j}__t{DEFAULT_THRESHOLDS[j]}.jsonl'))
    sh(f'python scripts/mechanism.py --run_dir {RUN} --judges {judges} --cooccurrence {COOC}')
    sh(f'python scripts/mechanism.py --run_dir {EVAL_RUN} --judges yolov8x mask2former --cooccurrence {COOC}')
"""),
    md("""
## What to send back

The printed tables from steps A or G, F, I and J. They are also saved on Drive:
- `runs/bon_pilot/<model>/overopt/curves__gold-<gold>.json` and `.png`
- `labels/picks_report.json`
- `runs/*/mechanism/summary.json`
"""),
]


LABEL = [
    md("""
# IOQM: label counts

Needs **no GPU**: a standard CPU runtime is enough, so it costs no compute credits.

Run all cells. The labeller opens below the last cell.

**How to label**
1. Pick a class on the right, then click each instance of it in the image. A numbered dot marks every click; **Undo click** removes the last one.
2. The count boxes fill in from your clicks. You can also type a count directly.
3. Count everything you can see, including partly hidden instances.
4. Tick **Ambiguous** when a count is genuinely unclear, e.g. objects fused together or a class you can't tell. These images are kept but left out of the ground truth.
5. **Skip** only if the image is unusable.

You only see the class names, not what was asked for or what the counters found. That's on purpose, so the labels aren't anchored.

Keys: **1/2/3** pick the class, **z** undoes a click, **a** toggles Ambiguous, **Enter** submits.

On a Mac with Google Drive for desktop you can skip Colab entirely: `python scripts/label.py` (or `--round picks`) opens the same labeller in your browser.

Progress is saved after every image. Close the tab any time; next session starts at the first unlabelled image. Resubmitting an image later overwrites the earlier label.

**Which queue:** two labelling rounds, each with its own labels file. Set both lines together.
- Round 1 (counter validation): `queue_v1.jsonl` → `labels.jsonl`
- Round 2 (best-of-k picks): `queue_picks.jsonl` → `labels_picks.jsonl`
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
DATA = '/content/drive/MyDrive/ioqm_data'

QUEUE, LABELS = f'{DATA}/labels/queue_v1.jsonl', f'{DATA}/labels/labels.jsonl'
# QUEUE, LABELS = f'{DATA}/labels/queue_picks.jsonl', f'{DATA}/labels/labels_picks.jsonl'

LABELER = 'labeler_1'      # change if someone else labels, so their labels can be told apart
"""),
    code("""
%cd /content
!rm -rf /content/ioqm && git clone -q -b phase0-harness https://github.com/semajyllek/ioqm.git /content/ioqm
%cd /content/ioqm
!pip install -q gradio
"""),
    code("""
from ioqm.labeler import build_app, launch
app = build_app(QUEUE, LABELS, LABELER)
launch(app)          # launch(app, share=True) gives a public link, e.g. to label on a tablet
"""),
]


if __name__ == "__main__":
    write(notebook(OVEROPT), ROOT / "notebooks" / "overopt_colab.ipynb")
    write(notebook(LABEL, gpu=False), ROOT / "notebooks" / "label_colab.ipynb")
