# Gold-judge over-optimization curves (klein-4B pilot, k ≤ 64)

*2026-11-06. 480 prompts × 64 seeds, FLUX.2-klein-4B. Gold judge: Qwen3-VL-8B (counts only), never used for selection. Unbiased best-of-k estimator over all 64 seeds; 95% cluster-bootstrap CIs over prompts. Source: `runs/bon_pilot/flux2_klein_4b/overopt/curves__gold-qwen3_vl_8b.json`, `mechanism/summary.json`; human agreement recomputed locally from `labels/labels.jsonl`.*

## 1. The gold judge is valid at k = 1

Agreement with one human labeller on the stratified `queue_v1` sample, re-weighted to the run:

| Run | Counter | Accuracy | κ | Precision of "exact" | Recall | Counter's count accuracy | Human's |
|---|---|---|---|---|---|---|---|
| pilot (125 imgs) | YOLOv8x | 0.93 | 0.80 | 0.89 | 0.81 | 0.23 | 0.25 |
| | Mask2Former | 0.94 | 0.83 | 0.92 | 0.83 | 0.23 | 0.25 |
| | **Qwen3-VL-8B** | **0.98** | **0.95** | 0.96 | 0.97 | 0.26 | 0.25 |
| eval (148 imgs) | YOLOv8x | 0.89 | 0.74 | 0.90 | 0.76 | 0.30 | 0.36 |
| | Mask2Former | 0.90 | 0.77 | 0.93 | 0.77 | 0.30 | 0.36 |
| | **Qwen3-VL-8B** | **0.97** | **0.92** | 0.94 | 0.97 | 0.37 | 0.36 |

The VLM judge is essentially unbiased on the headline number; both detectors undercount and understate accuracy. Open question: does the judge stay this good on images *selected* by a proxy? (`queue_picks`, 136 images, answers it.)

## 2. Curves (all 480 prompts)

| k | KL | Gold, YOLO picks | YOLO self | Gold, M2F picks | M2F self | Gold, ensemble picks | Ens. self | Gold oracle |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 0.273 | 0.232 | 0.273 | 0.225 | 0.273 | 0.180 | 0.273 |
| 4 | 0.64 | 0.413 | 0.438 | 0.403 | 0.424 | 0.421 | 0.345 | 0.486 |
| 16 | 1.83 | 0.544 | 0.661 | 0.517 | 0.647 | 0.567 | 0.535 | 0.707 |
| 64 | 3.17 | 0.629 [0.585, 0.673] | 0.835 | 0.588 [0.546, 0.629] | 0.821 | **0.669** [0.627, 0.710] | 0.704 | 0.869 |

- Best-of-k against a detector **does improve true counting** (0.27 → 0.59–0.67 at k = 64), so the answer to the title question is "both".
- But each single detector **overstates its own gain**: its self-score rises 0.60 while gold rises 0.32–0.36; the gap goes from −0.04 at k = 1 to **+0.21 (YOLO) / +0.23 (M2F) at k = 64**, and is still widening.
- The **agreement ensemble** is both the best selector by gold (0.669) and nearly calibrated (gap −0.09 → +0.04).
- Proxies capture only part of the available headroom: the gold oracle reaches 0.869 at k = 64.

## 3. By number of types

| | k | Gold, YOLO | YOLO self | Gold, M2F | M2F self | Gold, ens. | Ens. self |
|---|---|---|---|---|---|---|---|
| 1 type (60 prompts) | 16 | 0.846 | 0.903 | 0.825 | 0.846 | 0.851 | 0.803 |
| | 64 | 0.833 | 0.967 | 0.883 | 0.933 | 0.850 | 0.883 |
| 2 types (180) | 64 | 0.628 | 0.883 | 0.578 | 0.817 | 0.639 | 0.739 |
| 3 types (240) | 64 | 0.579 | 0.767 | 0.521 | 0.796 | 0.646 | 0.633 |

- Single-type prompts show the classic over-optimization shape: gold flattens after k ≈ 16 (YOLO picks 0.851 at k = 32, 0.833 at k = 64, within CI) while the proxy's self-score keeps climbing to 0.97.
- The ensemble's advantage is largest on 3-type prompts (0.646 vs 0.579 / 0.521) and stays calibrated there.

## 4. Mechanism under the gold judge (multi-type prompts)

- Per-type errors are **independent**: joint / product = 1.01 (2 types; p = 0.02, effect +0.003) and 1.00 (3 types; p = 0.64). Under the gold judge the detectors' pilot result holds.
- **No instance trading**: mean within-prompt correlation of signed errors 0.007 (2 types) and −0.013 (3 types).
- What predicts failure (log-odds per SD, SEs clustered by prompt): **total instances −1.29 [−1.58, −0.99]**; number of types −0.09, largest count −0.15, held-out pair −0.00, COCO min-PMI −0.03, all n.s. Same as with the detectors, so the "crowding" result is not just a detector artefact (the human-labelled check is still worth adding).

## Next

1. Human labels on `queue_picks` (136 images: each proxy's real pick at k = 1 and k = 64 for 50 prompts) → `picks_report.py`. This tests whether the gold judge itself is fooled by selection.
2. Second generator for generality.

---

## 5. Humans on the real picks (`queue_picks`, 2026-11-06)

50 random prompts; each proxy's actual pick among seeds 0..k−1 at k = 1 and k = 64; 136 distinct images, one labeller, blind to all verdicts. 27 flagged ambiguous (mostly partial or background objects and fused instances; 24% of k = 1 images, 17% of k = 64 images, so ambiguity does **not** rise under selection). `labels/picks_report.json`.

Accuracy of the pick, ambiguous images excluded (n = 38–44 prompts per row):

| Proxy | k | Human | Gold judge | Proxy's own verdict |
|---|---|---|---|---|
| any (same image) | 1 | 0.11 [0.03, 0.21] | 0.18 | 0.05–0.08 |
| YOLOv8x | 64 | **0.56** [0.41, 0.72] | 0.56 | **0.77** |
| Mask2Former | 64 | **0.53** [0.35, 0.68] | 0.55 | **0.80** |
| Ensemble | 64 | **0.61** [0.46, 0.75] | 0.57 | 0.59 |

Sensitivity, all 50 prompts: with ambiguous images counted as wrong, human accuracy at k = 64 is 0.44 / 0.42 / 0.54 (YOLO / M2F / ensemble) against self-verdicts 0.74 / 0.82 / 0.56; using the counts as entered it is 0.50 / 0.48 / 0.58. The conclusion is the same under every handling.

Readings:
- **Humans confirm the headline**: a single detector's picks at k = 64 are right about 55% of the time while the detector claims about 80%. The ~0.2–0.3 gap is real, not an artefact of the gold judge.
- **The gold judge tracks humans on the picks** (within 0.05 at k = 64), so selection pressure is not fooling it on average. Image-level, it agrees with the human on 57/71 usable k = 64-only images (80%; 7 false "exact", 7 false "wrong"), noisier than on the random sample but unbiased.
- **The ensemble is calibrated under pressure**: its self-verdict 0.59 vs human 0.61.
- At k = 1 the gold judge reads 0.18 vs human 0.11 (7 vs 4 of 38): small n, worth watching.
