# Draft: introduction and results outline

*Working draft, 2026-11-04. Numbers marked **[pilot]** come from the klein-4B best-of-16 pilot with a COCO detector as judge (`docs/results/2026-11-04_pilot_curves_and_mechanism.md`). They are placeholders until the gold-judge and human results are in. **[TBD]** marks results not yet run.*

---

## Working titles

1. *Counting on the Counter: How Much of Count-Control Progress Is Real?*
2. *Optimizing Against Counters: Over-optimization in Count-Faithful Text-to-Image Generation*
3. *Your Counter Is Being Fooled: Best-of-N and Reward Over-optimization for Object Counts*

## Abstract (sketch)

Text-to-image models still fail to render requested object counts, especially when a prompt asks for several object types at once. The dominant remedies, best-of-N selection and reinforcement learning with counting rewards, optimize the generator against an automatic counter, and progress is then measured with an automatic counter too. We ask how much of the resulting gain is real. Using a combinatorial benchmark of joint multi-type counting prompts, an independent gold judge that is never optimized against, and human labels of the selected images, we measure how accuracy under the proxy counter and under the gold judge diverge as optimization pressure grows. At best-of-16, the counter doing the selecting already rates its own picks about 0.2 higher than an independent counter does **[pilot]**, and the gap widens with k **[TBD: gold judge, k ≤ 64, human-confirmed]**. We find that per-type errors are close to independent and that failure is governed by the total number of requested instances rather than by the number of object types or how often the classes co-occur in real images **[pilot; TBD under gold judge]**. We release the benchmark, code, and the first set of human counts on generated images.

---

## 1. Introduction

**Paragraph 1: the problem.** Asking a text-to-image model for "three cups and two dogs" remains a reliable way to make it fail. Counting is among the weakest skills of current generators: on multi-category benchmarks every model is off by more than one object on average (M3T2IBench, 2510.23020), the best system on GeckoNum's multi-type prompts gets the exact numbers right 45% of the time (2406.14774), and accuracy collapses as the number of requested objects grows (NumBench, 2608.28206). Counts matter because they are one of the few prompt attributes with an objectively right answer.

**Paragraph 2: how the field fixes it, and the catch.** The strongest recent gains come from optimizing the generator against an automatic counter. Flow-GRPO raises SD3.5-M's GenEval counting accuracy from 0.50 to 0.95 with a detector-based reward (2505.05470); ABACUS trains against a count reward (2606.23835); best-of-N and reflection methods select or revise images with a detector or VLM in the loop (2501.09732, 2503.12271). Progress is then reported by an automatic counter as well, often the same kind of detector. When the optimizer and the measuring instrument share failure modes, a rising score can mean the generator has learned to produce images the counter miscounts rather than images with the right counts. This is the familiar pattern of reward over-optimization (Gao et al., 2022), but it has not been measured for counting, where, unusually, the ground truth is objective and a human can check it.

**Paragraph 3: what we do.** We measure it. We generate 64 images per prompt for a stratified set of joint multi-type counting prompts and let each of several proxy counters (two COCO detectors and their agreement ensemble) select the best of k, for k from 1 to 64. An independent gold judge, a vision-language model that is never used for selection and whose agreement with human counts we measure, scores the selections. An unbiased best-of-k estimator turns the 64 samples into a full accuracy-versus-pressure curve. Finally, humans label the proxies' actual picks at k = 1 and k = 64, so the central claim rests on human judgment, not on another model.

**Paragraph 4: what we find (fill in when results are in).**
- The proxy's estimate of its own picks pulls away from the gold judge's as k grows: +0.22 at k = 16 **[pilot]**, **[TBD at k = 64]**. On single-type prompts the judged accuracy plateaus after k = 4 while the proxy's own score keeps rising **[pilot]**.
- Humans **[TBD: confirm / partly confirm]** the gold judge's verdicts on the picks.
- Agreement ensembles **[TBD: delay / do not delay]** the divergence.
- Per-type errors are close to independent, so multi-type prompts are hard because per-type error rates compound; failure is driven by the total number of requested instances, not by the number of types or by how often the classes co-occur in COCO **[pilot; TBD under gold judge, with the crowding confound addressed]**.

**Paragraph 5: contributions.**
1. The first measurement of over-optimization for count-faithful generation, with human ground truth on the selected images.
2. A reusable protocol: disjoint proxy and gold judges, an unbiased best-of-k estimator, and a stratified human check that re-weights to the full distribution.
3. Mechanism results on joint counting: independence of per-type errors and the dominant role of total instance count.
4. Released artifacts: a combinatorial joint-counting prompt set with held-out class combinations, code, and human counts on generated images.

---

## 2. Related work (skeleton)

- **Counting benchmarks for T2I.** GenEval (2310.11513), T2I-CompBench++ (2307.06350), GeckoNum (2406.14774), HRS-Bench (2304.05390), GenEval 2 (2512.16853), M3T2IBench (2510.23020), NumBench (2608.28206), T2ICountBench, CoCoCount. Position: we use a benchmark as an instrument; our contribution is the optimization study.
- **Verifier reliability.** "Verifier Choice is a Benchmark Design Variable" (NeurIPS 2026) shows that verifier choice moves static scores by tens of points. We study what happens when generation is optimized against the verifier, which static audits cannot show.
- **Count-control methods.** Attention guidance (Counting Guidance, CountCluster, ATHENA, Bounded Attention), per-token optimization (2408.11721), layouts plus self-correction (LMD, SLD), RL (Flow-GRPO, ABACUS, Qwen-Image RL), inference-time search (2501.09732, Reflect-DiT).
- **Reward over-optimization.** Gao et al. 2022 for language models; T2I reward hacking (2601.03468, GRPO-Guard 2510.22319, GDRO 2601.02036). Our setting adds objective, human-checkable ground truth.
- **Counting models.** CountGD/CountGD++, SAM 3, multi-class counting failures (PrACo, PairTally); VLM counting (CountQA).

---

## 3. Results outline

Each subsection lists the claim, the evidence, and its status.

### 3.1 Setup

- Prompts: 480-prompt pilot, stratified by split and number of types (1, 2, 3), counts 1–5, COCO classes.
- Generator: FLUX.2-klein-4B, 64 seeds per prompt **[TBD: second generator, e.g. z_image_turbo, for generality]**.
- Proxies: YOLOv8x, Mask2Former, agreement ensemble. Gold: Qwen3-VL **[8B or 32B, TBD]**.
- Metric: exact count accuracy (all types exact), with MAE and signed bias as diagnostics.
- Estimator: unbiased best-of-k over all seeds; KL = log k − (k−1)/k as the pressure axis; cluster bootstrap over prompts.

### 3.2 The instrument: how good are the judges? (Table 1)

- Claim: the gold judge agrees with humans better than either proxy does.
- Evidence: counter-vs-human agreement on the stratified `queue_v1` sample (accuracy, κ, precision of a "correct" verdict, count bias), re-weighted to the run. Also the 2023 human labels.
- Status: **[TBD]** (needs `queue_v1` labels and gold scores).

### 3.3 Over-optimization curves (Figure 1, the headline)

- Claim: the proxy's own estimate rises faster than gold accuracy, and the gap widens with k.
- Evidence: gold accuracy and proxy self-score vs KL, one line pair per proxy, k = 1…64; gold-judge oracle (pass@k) as the ceiling.
- Pilot: gap +0.22 / +0.18 at k = 16 in the two detector-vs-detector directions **[pilot]**.
- Status: **[TBD]** full curve with gold judge.

### 3.4 Humans confirm the picks (Table 2)

- Claim: human verdicts on actual picks track the gold judge, not the proxy.
- Evidence: for 50 prompts, human / gold / proxy accuracy of each proxy's pick at k = 1 and k = 64.
- Failure mode worth reporting either way: if the gold judge is also fooled, humans show it.
- Status: **[TBD]** (`queue_picks` labels).

### 3.5 Which proxies are safer? (Figure 2)

- Claim: the agreement ensemble **[TBD: delays / does not delay]** divergence; proxy quality (from 3.2) predicts where gold accuracy peaks.
- Evidence: the curves of 3.3 split by proxy; gap at fixed KL versus proxy precision.
- Status: **[TBD]**.

### 3.6 Where the gap comes from (Figure 3)

- Claim: divergence is concentrated in specific strata, e.g. large counts or crowded scenes.
- Evidence: curves by number of types, largest count, total instances; gold vs proxy disagreement categories (overcount, undercount, fused objects from human "ambiguous" flags).
- Pilot: plateau on single-type prompts after k = 4 **[pilot]**.
- Status: **[TBD]**.

### 3.7 Mechanism: why joint counting is hard (Table 3, Figure 4)

- Claim A: per-type errors are near-independent (joint ≈ product of marginals; ratio 0.97–1.02, n.s.) **[pilot]**.
- Claim B: little evidence of instance trading between types (correlations −0.04 to 0.00) **[pilot]**.
- Claim C: total instances dominates failure; number of types, largest count and COCO co-occurrence add little **[pilot, both runs]**.
- Required before claiming: re-run under the gold judge and on human-labelled images, because detectors also miscount crowded scenes.
- Status: pilot done; **[TBD]** gold and human.

### 3.8 (If time) RL shows the same divergence

- Claim: training against a count reward shows the same proxy-gold divergence over training steps.
- Evidence: Flow-GRPO-style fine-tuning of klein-4B or SD3.5-M with a proxy reward, gold accuracy tracked every N steps; image-quality guard.
- Status: deferred; needs a stochastic-sampler pilot.

---

## 4. Limitations to state up front

- One labeller for most labels; a second annotator on a subset gives inter-rater agreement **[TBD]**.
- Gold judge is a model; humans check it on a stratified sample and on the picks, not on every image.
- COCO classes and counts ≤ 5; COCO co-occurrence is a proxy for the generators' training data.
- Best-of-k is the main optimizer; RL results are **[TBD / out of scope]**.

## 5. Figure and table checklist

| Item | Content | Source | Status |
|---|---|---|---|
| Fig. 1 | Gold vs proxy accuracy vs KL, per proxy | `overopt_curve.py` | pilot only |
| Fig. 2 | Gap vs proxy precision | `overopt_curve.py` + `agreement_report.py` | TBD |
| Fig. 3 | Curves by stratum | `overopt_curve.py --by` | pilot only |
| Fig. 4 | Accuracy vs total instances, joint vs product | `mechanism.py` | pilot only |
| Tab. 1 | Judge agreement with humans | `agreement_report.py`, `human_check.py` | TBD |
| Tab. 2 | Human / gold / proxy on real picks | `picks_report.py` | TBD |
| Tab. 3 | Failure regression | `mechanism.py` | pilot only |
