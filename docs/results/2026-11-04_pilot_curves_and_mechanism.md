# Pilot results: best-of-k curves (k ≤ 16) and mechanism analyses

*2026-11-04. Preliminary: both judges are COCO detectors (YOLOv8x at 0.5, Mask2Former at 0.9). No VLM gold judge and no human labels yet. Every claim below needs confirming under the gold judge and against human labels.*

Data: FLUX.2-klein-4B on the 480-prompt pilot × 16 seeds (`runs/bon_pilot/flux2_klein_4b`), and z_image_turbo on the 2,985-prompt eval subset × 4 seeds (`runs/v3_eval/z_image_turbo`). Produced with `scripts/overopt_curve.py` and `scripts/mechanism.py`.

## Best-of-k, one detector picks, the other judges (unbiased estimator over all 16 seeds)

| Proxy → judge | k=1 | k=4 | k=16 | Proxy's own score at k=16 | Gap at k=16 |
|---|---|---|---|---|---|
| YOLOv8x → Mask2Former | 0.224 | 0.338 | 0.452 [0.408, 0.496] | 0.673 | +0.221 |
| Mask2Former → YOLOv8x | 0.231 | 0.348 | 0.452 [0.410, 0.498] | 0.627 | +0.175 |

The gap grows steadily with k in both directions. On single-type prompts with Mask2Former picking, the judged accuracy is flat from k=4 to k=16 (0.679 → 0.683) while the proxy's own score keeps rising (0.719 → 0.833): the first sign of a plateau.

## Are per-type errors independent? (within prompt, permutation null, 1,000 permutations)

| Judge | Types | P(all exact) | Product of per-type rates | Ratio | p |
|---|---|---|---|---|---|
| YOLOv8x | 2 | 0.247 | 0.243 | 1.02 | 0.17 |
| YOLOv8x | 3 | 0.125 | 0.129 | 0.97 | 0.13 |
| Mask2Former | 2 | 0.240 | 0.236 | 1.02 | 0.12 |
| Mask2Former | 3 | 0.134 | 0.135 | 0.99 | 0.51 |

Errors on different types look independent: joint accuracy ≈ product of per-type accuracies.

Trading (correlation of signed errors between types): −0.030 (p = 0.16) and −0.022 (p = 0.06) for YOLOv8x; +0.000 (p = 0.98) and −0.040 (p = 0.001) for Mask2Former, for 2 and 3 types. Small and inconsistent. Caution: a detector that confuses two requested classes produces negative correlation by itself.

## What predicts failure (logistic regression, multi-type prompts, z-scored features, SEs clustered by prompt)

| Feature | klein-4B pilot (YOLO / M2F) | z_image_turbo eval (YOLO / M2F) |
|---|---|---|
| total instances | −1.30 / −1.39 (p < 1e-14) | −1.12 / −1.07 (p < 1e-31) |
| number of types | −0.06 / +0.04 (n.s.) | +0.12 / +0.10 (p ≈ 0.01–0.03) |
| largest count | −0.07 / −0.12 (n.s.) | −0.06 / −0.15 (n.s. / p = 0.03) |
| held-out pair or triple | n.s. | n.s. |
| min pairwise COCO PMI | n.s. | n.s. |

Total instances dominates. Holding it fixed, more types (hence smaller per-type counts) is if anything easier. COCO co-occurrence does not predict failure. Held-out status is null, as it should be for models never trained on our splits.

**Confound to resolve:** detectors also miscount crowded scenes, so part of the total-instances effect may be counter error. The gold judge and human labels decide this.
