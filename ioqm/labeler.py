"""
Gradio labeller for human counts (run it from notebooks/label_colab.ipynb).

Design choices:
- Blind: the labeller sees the image and the class names only. Not the prompt,
  the requested counts or any counter's counts, so the label records what is in
  the image rather than confirming what was asked for.
- Click to count: pick a class, click each instance; a numbered dot marks it.
  The count boxes fill in from the clicks and can be edited directly.
- "Ambiguous" flags images where a count is genuinely unclear (fused or partial
  objects, unclear class). They are kept but excluded from ground truth.
- Append-only JSONL with resume: the last record for an item wins, so a
  correction is just another submit.

Pure functions (testable without Gradio) come first; build_app() wires the UI.
"""

# no `from __future__ import annotations`: Gradio reads the gr.SelectData annotation at runtime

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .labels import read_labels

MAX_TYPES = 3
COLORS = ("#e6194b", "#3cb44b", "#4363d8")   # one per class slot


def load_queue(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def next_index(queue: Sequence[Mapping], done_ids: Sequence[str], start: int = 0) -> Optional[int]:
    done = set(done_ids)
    for i in list(range(start, len(queue))) + list(range(0, start)):
        if queue[i]["item_id"] not in done:
            return i
    return None


def add_point(points: Dict[str, List[Tuple[int, int]]], history: List[str], cls: str, xy: Tuple[int, int]):
    points.setdefault(cls, []).append((int(xy[0]), int(xy[1])))
    history.append(cls)


def undo(points: Dict[str, List[Tuple[int, int]]], history: List[str]):
    if history:
        cls = history.pop()
        if points.get(cls):
            points[cls].pop()


def point_counts(classes: Sequence[str], points: Mapping[str, Sequence]) -> List[int]:
    return [len(points.get(c, [])) for c in classes]


def make_record(item: Mapping, counts: Optional[Mapping[str, int]], ambiguous: bool, skipped: bool,
                notes: str, points: Mapping[str, Sequence], labeler: str) -> dict:
    return {
        "item_id": item["item_id"], "run_dir": item["run_dir"], "image": item["image"],
        "counts": None if skipped else {k: int(v) for k, v in (counts or {}).items()},
        "ambiguous": bool(ambiguous), "skipped": bool(skipped), "notes": notes or "",
        "points": {k: [list(p) for p in v] for k, v in points.items()},
        "labeler": labeler, "ts": time.time(),
    }


def append_label(path: Path, rec: Mapping) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def render(image_path: str, classes: Sequence[str], points: Mapping[str, Sequence]):
    from PIL import Image, ImageDraw
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = max(6, img.width // 90)
    for slot, c in enumerate(classes):
        for i, (x, y) in enumerate(points.get(c, []), 1):
            draw.ellipse([x - r, y - r, x + r, y + r], fill=COLORS[slot], outline="white", width=2)
            draw.text((x + r + 2, y - r), str(i), fill=COLORS[slot])
    return img


def image_path(item: Mapping) -> str:
    from .generate import images_dir
    return str(images_dir(Path(item["run_dir"])) / item["image"])


def build_app(queue_path: Path, labels_path: Path, labeler: str):
    import gradio as gr
    from .classes import prompt_noun

    queue = load_queue(queue_path)

    def fresh(idx: Optional[int]) -> dict:
        return {"idx": idx, "points": {}, "history": []}

    def view(st):
        """Outputs for the current item: image, radio, 3 numbers, ambiguous, notes, progress."""
        done = read_labels(labels_path)
        n_done = sum(1 for it in queue if it["item_id"] in done)
        if st["idx"] is None:
            msg = f"**All {len(queue)} items labelled.** Thank you."
            hidden = [gr.update(visible=False)] * MAX_TYPES
            return [None, gr.update(choices=[], value=None), *hidden, False, "", msg]
        item = queue[st["idx"]]
        classes = item["classes"]
        nouns = [prompt_noun(c, 2) for c in classes]
        counts = point_counts(classes, st["points"])
        nums = [gr.update(visible=True, label=f"{nouns[i]}  ({['red', 'green', 'blue'][i]})", value=counts[i])
                if i < len(classes) else gr.update(visible=False, value=0) for i in range(MAX_TYPES)]
        msg = (f"Item {st['idx'] + 1} of {len(queue)}  ·  labelled {n_done}  ·  "
               f"count every **{' / '.join(nouns)}** you can see, including partly hidden ones.")
        return [render(image_path(item), classes, st["points"]),
                gr.update(choices=nouns, value=nouns[0]), *nums, False, "", msg]

    def on_click(st, active_noun, evt: gr.SelectData):
        if st["idx"] is None or active_noun is None:
            return st, gr.update(), *([gr.update()] * MAX_TYPES)
        item = queue[st["idx"]]
        cls = item["classes"][[prompt_noun(c, 2) for c in item["classes"]].index(active_noun)]
        add_point(st["points"], st["history"], cls, evt.index)
        return st, render(image_path(item), item["classes"], st["points"]), *_nums(st)

    def _nums(st):
        classes = queue[st["idx"]]["classes"]
        counts = point_counts(classes, st["points"])
        return [gr.update(value=counts[i]) if i < len(classes) else gr.update() for i in range(MAX_TYPES)]

    def on_undo(st):
        if st["idx"] is None:
            return st, gr.update(), *([gr.update()] * MAX_TYPES)
        undo(st["points"], st["history"])
        item = queue[st["idx"]]
        return st, render(image_path(item), item["classes"], st["points"]), *_nums(st)

    def on_clear(st):
        if st["idx"] is None:
            return st, gr.update(), *([gr.update()] * MAX_TYPES)
        st["points"], st["history"] = {}, []
        item = queue[st["idx"]]
        return st, render(image_path(item), item["classes"], st["points"]), *_nums(st)

    def save_and_next(st, skipped, ambiguous, notes, *nums):
        if st["idx"] is not None:
            item = queue[st["idx"]]
            counts = {c: int(nums[i] or 0) for i, c in enumerate(item["classes"])}
            append_label(labels_path, make_record(item, counts, ambiguous, skipped, notes, st["points"], labeler))
        done = read_labels(labels_path)
        nxt = next_index(queue, list(done), start=(st["idx"] or 0) + 1)
        st = fresh(nxt)
        return [st, *view(st)]

    with gr.Blocks(title="IOQM counts") as app:
        st0 = fresh(next_index(queue, list(read_labels(labels_path))))
        state = gr.State(st0)
        progress = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(type="pil", interactive=False, show_label=False, height=640)
            with gr.Column(scale=1):
                active = gr.Radio(label="Clicking adds a", choices=[])
                nums = [gr.Number(precision=0, minimum=0) for _ in range(MAX_TYPES)]
                with gr.Row():
                    b_undo = gr.Button("Undo click")
                    b_clear = gr.Button("Clear clicks")
                amb = gr.Checkbox(label="Ambiguous (fused / partial / unclear class)")
                notes = gr.Textbox(label="Notes (optional)", lines=2)
                b_submit = gr.Button("Submit", variant="primary")
                b_skip = gr.Button("Skip (image unusable)")
        view_out = [img, active, *nums, amb, notes, progress]
        app.load(lambda s: view(s), inputs=state, outputs=view_out)
        img.select(on_click, inputs=[state, active], outputs=[state, img, *nums])
        b_undo.click(on_undo, inputs=state, outputs=[state, img, *nums])
        b_clear.click(on_clear, inputs=state, outputs=[state, img, *nums])
        b_submit.click(lambda s, a, n, *v: save_and_next(s, False, a, n, *v),
                       inputs=[state, amb, notes, *nums], outputs=[state, *view_out])
        b_skip.click(lambda s, a, n, *v: save_and_next(s, True, a, n, *v),
                     inputs=[state, amb, notes, *nums], outputs=[state, *view_out])
    return app
