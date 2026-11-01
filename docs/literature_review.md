# IOQM literature review: counting objects, and generating counted objects

*Compiled 2026-11-01. Each claim was checked against the arXiv page, PDF, repository, or leaderboard it cites. The full per-paper tables, and the lists of claims that could **not** be verified, are in [`lit_review_sources/`](lit_review_sources/):*
- *[counting_and_segmentation.md](lit_review_sources/counting_and_segmentation.md)*
- *[count_controlled_generation.md](lit_review_sources/count_controlled_generation.md)*

*Treat any number not cited here as unverified. Three load-bearing claims were also spot-checked by hand: the M3T2IBench and NumBench arXiv pages, and the SD3.5-M GenEval overall result of 0.63 → 0.95 reported in Flow-GRPO.*

IOQM involves two separate problems:

- **Task A, perception.** Given an image and a list of object classes, count the instances of each class.
- **Task B, generation.** Given a prompt such as "3 A and 2 B and 4 C", produce an image with exactly those counts.

Task A is also the measuring instrument for Task B. Any claim about Task B can only be as good as the counter used to score it.

---

## 1. Where IOQM sits among T2I counting benchmarks

IOQM is **not** the only benchmark with joint multi-class counts. At least six others put two or more object types, each with its own count, in a single prompt:

| Benchmark | arXiv | Joint counts? | Classes / counts | Scorer | Human-validated scorer? |
|---|---|---|---|---|---|
| GenEval (NeurIPS'23 D&B) | 2310.11513 | **No.** `counting` is one class with N∈{2,3,4}; `two_object` checks presence only | COCO-80 | Mask2Former | Yes. Counting κ rises from 0.37 to 0.65 when the threshold is raised to 0.9 |
| T2I-CompBench++ numeracy (TPAMI'25) | 2307.06350 | Yes, ~40% of prompts are multi-object | ~150 nouns, counts 1–8 | UniDet with partial credit | τ = 0.43 |
| GeckoNum (NeurIPS'24 D&B) | 2406.14774 | Yes: 2- and 3-additive prompts, 100 each | ~40 nouns, counts 1–5 | Human annotators only | Human by design. Best exact-number accuracy is 45.2% (DALL·E 3) |
| HRS-Bench (ICCV'23) | 2304.05390 | Yes, 2 types | LVIS classes, rewritten by an LLM into natural scenes | UniDet, F1 | r = 0.93 |
| GenEval 2 (Dec'25) | 2512.16853 | Possible, 1–3 objects with counts 2–7, mixed with attributes and relations | 40 non-COCO objects | Soft-TIFA (Qwen3-VL) | Yes, AUROC 94.5 |
| **M3T2IBench** (Oct'25) | 2510.23020 | Yes, a count for each category, at most 5 instances in total | COCO (66 classes) | Mask2Former | r = 0.67 |
| **NumBench** (Aug'26) | 2608.28206 | Yes, 2 categories | 1,600 categories, counts 1–100 | Ensemble of 3 detectors | κ = .82 at counts 1–25 |

Single-class counting benchmarks include T2ICountBench (counts 1–15, no model averages above 50%) and CoCoCount (from Make It Count).

### How each benchmark scores counts

Benchmarks disagree less about *what* to count than about *how to score* a near miss. Every scheme below sits somewhere between two extremes: all-or-nothing exact match, and graded credit for partial correctness. Who does the counting also differs: a closed-set detector, an open-vocabulary detector ensemble, a VLM, or humans.

| Benchmark | Who counts | Unit scored | Scoring rule |
|---|---|---|---|
| GenEval `counting` | Mask2Former (threshold 0.9 for counting) | image, one class | Pass/fail: the detected count equals N |
| GenEval++ | GPT-4.1 checklist | image | Pass only if every condition in the prompt holds |
| T2I-CompBench++ numeracy | UniDet | each object, averaged | Each of n objects earns 1/(2n) for being detected and another 1/(2n) for the exact count |
| GeckoNum | 5 human annotators (mode count) | each noun | Exact-number accuracy for each noun |
| HRS-Bench counting | UniDet | image | Precision / recall / F1 on counts (exact formula not verified) |
| M3T2IBench | Mask2Former (0.3, NMS IoU 0.9) | image | Count bias Σₖ\|nₖ − mₖ\|, the total instance error, inside a combined AlignScore |
| NumBench | 3-detector ensemble with count-dependent thresholds | image | cw-NPS, a confidence-weighted numeric precision score |
| GenEval 2 | Qwen3-VL-8B (Soft-TIFA) | each atom | Mean (arithmetic or geometric) of the VLM's P(yes) for each atom, e.g. "are there 3 dogs?" |
| TIFA / DSG / VQAScore | VQA model | question / prompt | Count questions answered by VQA. VQAScore gives a single holistic P(yes) |
| T2ICountBench | 5 human graders | prompt | Pass if any of 4 samples has the right count |
| CoCoCount (Make It Count) | YOLOv9, and humans | image, one class | Exact-count accuracy |
| **Ours (v3)** | Calibrated detector, validated against human labels (Phase 1) | image, plus each object | **Count accuracy** (all types exact) as the headline. Diagnostics: per-object count accuracy, count MAE, signed count bias, extra-object rate |

**Worked example.** Prompt: "2 cats and 2 dogs". The image has **3 cats and 2 dogs**.

| Scheme | Score | Why |
|---|---|---|
| Exact match (GenEval-style; ours, headline) | **0** | The cat count is wrong |
| T2I-CompBench++ | **0.75** | Cats: ¼ for being detected + 0 (count wrong). Dogs: ¼ + ¼ |
| GeckoNum-style, per noun | **0.5** | 1 of 2 nouns has the exact count |
| M3T2IBench count bias | **1** (lower is better) | \|3−2\| + \|2−2\| |
| Ours: per-object count accuracy / MAE / bias | **0.5 / 0.5 / +0.5** | Diagnostics: one of two types right; off by 0.5 objects per type on average; overgenerates |

The same image scores anywhere from 0 to 0.75 depending on the benchmark. That's why numbers from different benchmarks can't be compared directly, and why this project re-runs every baseline under one fixed scheme.

**Why exact match is our headline.** "Generate 2 cats and 2 dogs" is satisfied only if both counts are right. Partial-credit schemes reward images that a user would reject. Exact match also extends GenEval's established counting metric to 2–3 types, so the numbers are easy to interpret. The diagnostics stay because they show *how* a model fails: over- or under-generation, dropped types, or extra objects. Those failure modes are what steering methods target.

**What IOQM adds.** No other benchmark combines all four of the following:
1. Exhaustive class × count enumeration: every pair and triple of classes, at every count from 1 to 5.
2. Unnatural co-occurrences included deliberately. HRS-Bench goes the other way and has an LLM rewrite prompts into plausible scenes.
3. COCO classes, so standard closed-set detectors can score it.
4. Up to 3 jointly counted types, scored for an exact match.

Point 1 is also the setup for a controlled study: does a generator's accuracy depend on how often the two classes co-occur in real data? The COCO co-occurrence matrix in `notebooks/ioqm_combo_correlation.ipynb` is exactly the tool for that question. None of the benchmarks above can isolate it.

**IOQM's main weakness** is the scorer. Every benchmark in the table except GeckoNum (which is scored by humans) has published how well its automatic scorer agrees with humans. IOQM has not. Its own human-labeled data shows large disagreement: the best detector, DETA-Swin, has a mean |Δ| of 0.24 on soft-IOQM. Two further critiques apply directly:
- **Echo-4o and GenEval 2:** detector-based scorers stop tracking human judgment as generators improve. GenEval's scorer rated Gemini 2.5 Flash Image at 75.4; humans rated it 93.1.
- **No 3-type results for count-control methods:** GeckoNum's 3-additive set (100 prompts, scored by humans) does report *base-model* accuracy on three-type joint prompts. But no count-control *method* (attention guidance, layout, RL) has been evaluated on 3-type joint prompts. That is the gap IOQM can fill.

**Bottom line:** IOQM is worth publishing as a benchmark of controlled, exhaustive joint counts that tests generalization to rare combinations. It is only credible once its scorer is validated against humans.

---

## 2. Task A: counting and segmentation (the instrument)

### State of the art by sub-area

- **Few-shot / zero-shot counting (FSC-147).**
  - Results: CountGD (NeurIPS'24, 2407.04619) and CountGD++ (CVPR'26, 2512.23351) are effectively tied, at roughly 5.7–5.9 MAE on the corrected test split with text plus exemplars. MixCount fine-tuning (2605.18063) brings CountGD++ down to 4.68 / 15.66.
  - Caveat: FSC-147 is a dense, high-count benchmark whose RMSE is dominated by outliers. Fixing a single test image moves CountGD's RMSE from 43.65 to 24.09. **It is the wrong yardstick for counts of 1–5.**
- **Small-count benchmarks** (closest to IOQM's range).
  - SAM 3 (2511.16719, Nov'25, gated open weights) leads the checked results. It scores CountBench MAE 0.12 / 93.8% accuracy and PixMo-Count 0.21 / 86.2% (SAM 3 Table 3).
  - That beats Molmo-72B (92.4% / 88.8%), Gemini 2.5 Pro (92.4% / 78.2%) and DINO-X (82.9% / 85.0%).
  - The CountBench comparison is close and the benchmark is near saturation; on PixMo-Count, Molmo is actually ahead. Newer VLM figures of 97–98% are vendor self-reports and were not verified.
- **COCO-80 closed-set detection.**
  - Co-DINO-Inst ViT-L (66.0 box AP / 57.1 mask AP on test-dev, open weights) is the strongest drop-in replacement for IOQM's DETR, DETA and YOLOv8x.
  - Mask2Former should also be run, because it is GenEval's scorer and keeps IOQM comparable to GenEval.
- **Open-vocabulary detectors.** Grounding DINO 1.5/1.6 and DINO-X are available only through an API, so they are not reproducible instruments. OWLv2 and Florence-2 are weaker.
- **VLM counting.** CountBench is saturated, but counting is far from solved. On CountQA (2508.06585), the best of 15 models scored 42.9% on dense real-world counting.
- **Multi-class counting** is the least-studied requirement and the one IOQM needs most.
  - PrACo (WACV'25, 2409.15953), PrACo++ (2605.02752) and PairTally (2509.13939) all show that counters often **count the wrong class** when two classes appear in one image.
  - The only checked fix is CountGD++ with negative prompts: PairTally MAE drops from 46.41 to 35.27, then to 28.87 with MixCount.
  - SAM 3 has **no published multi-class result.** Its behavior there is untested, not known to be bad.

### Counters on *generated* images

No paper checks a modern counter against human counts on T2I outputs. All existing human checks used detectors:
- **GenEval (Mask2Former):** raising the threshold to 0.9 lifted counting κ from 0.37 to 0.65. NMS did not help.
- **Make It Count (YOLOv9 vs human raters):** the gap depends on the generator. YOLOv9 scored DALL·E 3 13 points lower than humans did.
- **NumBench (3-detector ensemble):** κ falls from .82 at counts 1–25 to .33 at counts 76–100.

**No study measures how detectors handle fused or melded objects,** which are the defining artifact of generated images. IOQM's ~600 human-labeled images would therefore be the first counter-vs-human benchmark on generated images. Extended to about 1–2k images with double annotation, they would make a publishable contribution on their own.

### Fine-tuning a counter for generated images

- **What has worked:** adding 1k Blender-rendered images (MixCount) cut CountGD++ MAE by about 20%, on both FSC-147 and PairTally.
- **What is missing:** nothing has been fine-tuned on human-labeled T2I outputs. That is an open gap.
- **Order of work:** threshold calibration comes first. GenEval found it was the single largest lever on agreement.

---

## 3. Task B: count-faithful generation

### Current leaderboard (GenEval counting: one class, N = 2–4)

GenEval counting numbers for the same model drift by about 0.05 to 0.14 between papers (FLUX.1-dev is reported at 0.74, 0.75 and 0.79). Treat the table as ordinal.

| Model | Open | GenEval overall | Counting | Source |
|---|---|---|---|---|
| SD1.5 | ✓ | 0.43 | 0.35 | Flow-GRPO Tbl 1 |
| SDXL | ✓ | 0.55 | 0.39 | BAGEL Tbl 5 |
| FLUX.1-dev | ✓ | 0.66 | 0.74 | Qwen-Image Tbl 4 |
| FLUX.2-klein-9B | ✓ | 0.85 | 0.83 | ERNIE-Image Tbl 4 |
| Z-Image | ✓ | 0.84 | 0.78 | Z-Image Tbl 9 |
| BAGEL | ✓ | 0.82 | 0.81 | BAGEL Tbl 5 |
| Qwen-Image | ✓ | 0.87 | 0.89 | Qwen-Image Tbl 4 |
| Qwen-Image-RL | base weights ✓ | 0.91 | 0.93 | Qwen-Image Tbl 4 |
| **SD3.5-M + Flow-GRPO**† | ✓ | **0.95** | **0.95** (from 0.50) | Flow-GRPO Tbl 1 |
| GPT Image 1 | ✗ | 0.84 | 0.85 | Qwen-Image Tbl 4 |
| Seedream 3.0 | ✗ | 0.84 | 0.91 | Qwen-Image Tbl 4 |

† Trained with RL on GenEval-template prompts (deduplicated against the test set) and a GenEval-style detector reward. This is in-distribution optimization, not a clean comparison with untuned models.

GenEval's counting task is saturated. Joint counting is not:
- **GenEval 2, count column:** Qwen-Image scores 70.3 and Gemini 2.5 Flash Image 70.1.
- **GenEval 2, by atomicity:** FLUX.1-dev drops to 0% above atomicity 6.
- **M3T2IBench:** every model is off by more than 1 instance on average.
- **Not found:** no primary-source GenEval numbers for FLUX.2-dev, Imagen 4, Nano Banana Pro, or GPT-Image-1.5/2.

### Methods, grouped by mechanism

**Training-free attention guidance**
- Counting Guidance (2306.17567); CountCluster (2508.10710), +18.5 pp count accuracy per its abstract; ATHENA (2603.19676).
- Bounded Attention (2403.16990): .641 on NumBench double-category prompts, the best among the open/specialized systems the review extracted.
- Make It Count / CountGen (CVPR'25, 2406.10210): CoCoCount accuracy 28 → 50 by YOLO, 26 → 54 by humans. It handles **single class only**, as the paper states.

**Per-token optimization**
- Zafar et al. (WACV'26, 2408.11721): accuracy up to 9 objects rises from 26.9% to 45.6%.

**Layouts and self-correction**
- LMD (2305.13655), single-class numeracy at counts 1–5 on 10 COCO classes: SD alone 39%, LMD 62%, LMD+ 86%. With perfect layouts the ceiling is 97%.
- SLD (2311.16090) adds detector-driven editing on top: LMD+ 82% → **98%**.
- This is the strongest training-free evidence, but it is on single-class prompts.

**Online RL**
- Flow-GRPO (2505.05470): SD3.5-M counting 0.50 → 0.95, using a rule-based detector reward with partial credit and KL β = 0.04.
- ABACUS (2606.23835, SIGGRAPH Asia'26): count-reward GRPO with reward exp(−|ĉ−c*|/c*). CoCoCount reaches **77% by human judgment, vs 54% for CountGen**. Single class only.
- Qwen-Image's own RL stage: counting 0.89 → 0.93.
- CompGen (2511.18378) uses cardinality as a curriculum axis.

**Inference-time search**
- Ma et al. (2501.09732): FLUX T2I-CompBench numeracy 0.617 → 0.679, with a warning about "verifier hacking".
- Reflect-DiT (2503.12271): SANA counting 0.58 → 0.80 with 20 samples, versus 0.73 for best-of-20.

**Reward hacking is documented, not hypothetical**
- Without KL, Flow-GRPO's image quality collapses: DeQA falls from 4.01 to 2.77.
- GDRO (2601.02036) shows "almost flat drawings" at high GenEval reward.
- GRPO-Guard (2510.22319) shows the proxy reward rising while real quality falls.
- Hong et al. (2601.03468) find that reward ensembles only partly help.

### Verdict

1. **Online RL with a per-type count reward on an open flow model.** This route has the largest verified gains. IOQM's combinatorial generator is a ready source of training prompts.
   - It requires a reward counter that differs from the evaluation counter, held-out class combinations, KL or an artifact regularizer, and human-validated final numbers.
2. **LLM layout, then layout-grounded generation, then detector self-correction (LMD+ / MIGC / Bounded Attention, followed by SLD).** This is the best training-free route and is cheap to try. It handles unnatural co-occurrences well, because the layout comes from the LLM rather than the image prior.
   - Unproven at 3 types with up to 15 instances. No count-control method has been tested on 3-type joint prompts.
3. **Verifier-guided best-of-N or reflection**, layered on either route. It is trivial to add.
   - Inflated by construction if the verifier is the evaluation counter.
