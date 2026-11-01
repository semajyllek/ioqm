# IOQM revival plan: a validated benchmark, then SOTA counter and generator

*Drafted 2026-11-01. Companion document: [literature_review.md](literature_review.md).*

## Goal and priority

**Primary goal:** a text-to-image method that is state of the art on joint multi-class counting prompts ("3 A and 2 B and 4 C"). The claim must hold on IOQM and on at least one external benchmark.

**Secondary goal, and a prerequisite:** a validated counter for generated images. That counter is the instrument for the primary goal, and it is potentially a contribution in its own right. No published work checks counters against human counts on T2I outputs, and none studies fused objects.

The order is fixed: **instrument, then baselines, then method**. Each phase ends with a go/no-go gate.

---

## Phase 0: Fix the harness (about 1–2 weeks)

### Bugs found by running the current code

| # | Bug | Example | Fix |
|---|---|---|---|
| 1 | `soft_ioqm` sums uncapped per-object ratios before clipping, so overcounting one object hides undercounting another | "2 cats and 2 dogs" with 4 cats and 0 dogs detected scores soft = 1.0 | Score each type as 1 − min(1, \|det−exp\|/exp), then average |
| 2 | Extra (hallucinated) classes are never penalized | "1 cat" with 7 dogs and 3 people detected scores 1.0 / 1.0 | Report a separate *extra-class rate*. Do not fold it into the count score |
| 3 | `check_object` uses prefix matching | "people" never matches the label "person", so all 176 v2 prompts with "people" score 0 for that object. "car" matches "carrot", and both classes are in v2 | Use an explicit canonical map: prompt noun → COCO id, with an adapter for each detector. DETR uses the 91-id scheme with N/A slots, and label strings differ between detectors |
| 4 | Bad plurals from TextBlob | "mouses", "sheeps", "broccolis" | Use a hand-written plural table for the 80 classes |
| 5 | One image per prompt, keyed by the prompt in the filename; no seeds | – | Store JSONL metadata (prompt_id, classes, counts, model, seed, sampler config). Use 4 seeds per prompt |
| 6 | *(Retracted after reading `run_gen.py`)* The "SDXL refiner" results are actually SDXL base plus refiner run together (80/20 split, 40 steps). They are only *named* after the refiner model ID | – | Label them "SDXL base+refiner" in reports |

Fixing bug 3 also fixes the hard metric for every "people" prompt.

### Reporting names

Report results in the field's terms, not as "IOQM". The headline is **count accuracy**, an exact-match rate in the style of GenEval's counting task, extended to 2–3 object types.

| Paper name | Code field |
|---|---|
| Count accuracy (headline) | `prompt_exact` |
| Per-object count accuracy | `type_exact` |
| Count MAE | `abs_err` |
| Count bias | `signed_err` |
| Extra-object rate | `extra_class` |

`type_soft` is a secondary diagnostic only.

### Metrics, fixed before any results come in

- **Primary:** prompt-level exact match, meaning every requested type has exactly the right count. This is stricter than the current hard IOQM and is the metric no other benchmark reports.
- **Secondary:**
  - per-type exact accuracy (the old hard IOQM)
  - per-type count MAE and signed bias (over- vs under-generation)
  - extra-class rate
  - per-type presence
- **Reporting rules:**
  - Stratify by number of types (1/2/3), by maximum count (1–5), and by class-pair co-occurrence quantile.
  - Give 95% bootstrap CIs, resampling prompts.
  - Compare models with paired tests (McNemar or paired bootstrap) on identical prompts and seeds.

### IOQM v3 prompt set

Keep v1 and v2 so old results can be reproduced. Then build v3:

- **Class selection.** Pick about 20–24 COCO classes stratified on two axes:
  - detectability, taken from Phase 1 per-class counter accuracy
  - co-occurrence frequency, from the COCO combo matrix already computed

  v1's 10 classes are mostly small and hard to detect (tie, toothbrush, mouse, ski, parking meter), which mixes up "the generator failed" with "the detector failed".
- **Structure.** Use nested exhaustive enumeration with counts 1–5. Fully exhaustive triples over 24 classes would be C(24,3)×125 ≈ 253k prompts, which is not feasible.

  | Stratum | Classes | Prompts |
  |---|---|---|
  | singles | all 24 | 24 × 5 = 120 |
  | pairs | all 24 | C(24,2) × 25 = 6,900 |
  | triples | a nested subset of 8 (spanning detectability and co-occurrence) | C(8,3) × 125 = 7,000 |
  | **total** | | **~14k prompts** |

- **Image budget.**
  - The full ~14k set (or its RL training portion) is used for training prompts, and for the co-occurrence analysis on 2–3 models.
  - Baselines are scored on a **stratified evaluation subset of ~3k prompts** (balanced over number of types, maximum count, and co-occurrence quantile), at 4 seeds = **~12k images per model**.
  - My rough estimate for FLUX/Qwen-Image-class models is ~3–5 s per image on an A100-class GPU, which is about **10–17 GPU-hours per model**, or **~100–170 GPU-hours** for 10 baselines. This is an estimate, to be measured in Phase 0.
  - For comparison, v1 at 4 seeds would be ~65k images per model.
  - 93% of v1 is 3-type prompts. The nested design and stratified subset correct that imbalance.
- **Splits.**
  - **test-ID:** class combinations and count tuples that are also seen in training.
  - **test-OOD:**
    - (a) held-out class pairs and triples, including the rarest COCO co-occurrences
    - (b) held-out count tuples (for example, every prompt where two types both have count 5)
  - **train:** prompt templates for RL only. Never evaluate on these.
- **Prompt-form sensitivity** (a small sweep): "and" vs commas, with and without "on a table", digits vs number words. The repo already has the start of this in `prompt_experiments.py`.

---

## Phase 1: The instrument (Task A, about 3–5 weeks)

### 1a. IOQM-Human: a counter-vs-human benchmark on generated images

- **Starting point:** the ~600 existing human-labeled images. Extend to **≥1,500 images** from at least 6 generators, spanning 2023 and 2025–26 models, because errors depend on the generator.
- **Annotation:**
  - **Double annotation.** Two annotators label each image, so inter-annotator agreement gives a human ceiling.
  - **Point clicks per instance** rather than bare counts. They cost about the same, allow localization-aware scoring, and become training data for 1c.
  - **A written rule for artifacts:** fused, partial and duplicated objects. Adopt NumBench's rule, "count visible bodies when separable". Annotators also flag an image as *ambiguous*.
- **Splits:** split by generator, so a held-out generator tests whether the counter generalizes.

### 1b. Benchmark candidate counters

| Candidate | Why | Notes |
|---|---|---|
| SAM 3 / 3.1 (one text prompt per class) | Best verified small-count results: CountBench 93.8%, PixMo-Count 86.2% | Gated weights. Untested on multi-class images |
| CountGD++ with the other prompt classes as negative prompts | Only verified fix for wrong-class counting (PairTally 46.4 → 35.3 MAE) | Open weights on Hugging Face |
| Co-DINO-Inst ViT-L | Strongest open COCO-80 instance segmenter | Closed-set, which fits IOQM |
| Mask2Former at GenEval's thresholds (0.3, and 0.9 for counting) | Comparability with GenEval and M3T2IBench | – |
| Qwen3-VL or Molmo, counting by pointing | VLM baseline; points can be audited | – |
| YOLOv8x, DETA-Swin (legacy) | Continuity with 2023 results | – |
| Ensemble (vote of the top 2–3) | NumBench-style robustness | – |

- **Calibration:** tune thresholds per class with 5-fold CV on IOQM-Human, following GenEval's protocol. GenEval showed calibration is the single largest lever on agreement.
- **Metrics:** per-type exact agreement, MAE, and κ against the human consensus, plus prompt-level agreement. Stratify by count, number of types, generator, and the ambiguous flag.

**Gate 1:** the best counter (or ensemble) must reach both:
- prompt-level agreement within 3 points of human–human agreement
- κ ≥ 0.8 on the held-out generator

If it passes, freeze it as the **evaluation counter** and continue. If it fails, go to 1c.

### 1c. (Only if Gate 1 fails) Fine-tune a counter for generated images

- **Candidates:** fine-tune CountGD++ or SAM 3 on the point-annotated training split of IOQM-Human. Optionally mix in synthetic data, in the style of MixCount's Blender renders (20% MAE gain on CountGD++).
- **Test:** on the held-out generators.
- **Data size:** the literature gives no data-size guidance for this domain. Plan for a few thousand labeled images, as an estimate.
- **Payoff:** passing Gate 1 here also yields the "SOTA counter on generated images" contribution.

### Instrument hygiene, applying from here on

Keep two counters with different architectures. Use one as the **evaluation counter** and the other as the **reward/verifier counter**. Never optimize against the evaluation counter.

---

## Phase 2: Baselines (Task B, about 2–3 weeks of compute)

Run on test-ID and test-OOD with 4 seeds and default sampler settings, logging all settings.

- **Open models:** SD1.5 and SDXL-base (for continuity), SD3.5-M/L, FLUX.1-dev, FLUX.2 (klein / dev), Qwen-Image, Z-Image, BAGEL, HiDream-I1.
- **Closed models, if there is budget:** GPT-Image, a Gemini image model, Seedream.
- **Training-free count methods:** LMD+, LMD+ followed by SLD, Bounded Attention, and best-of-N with the reward counter as selector (N = 4, 16).
- **External validity:** also score the top 3 models on GeckoNum (2/3-additive), T2I-CompBench++ numeracy, M3T2IBench, and GenEval counting.

### Analyses (each one a paper section)

1. **Accuracy against co-occurrence frequency.** Fit a mixed-effects logistic regression, with prompt-level exact match as the outcome and log COCO co-occurrence as the predictor. Control for per-class detectability, count, and number of types, with random effects for class. This is the rigorous version of the existing combo-correlation notebook.
2. **Error profile.** Plot the confusion between requested and generated count for each type. Does the model over- or under-generate? Does accuracy fall off sharply at 3 types?
3. **Prompt-form sensitivity.**

**Gate 2:** model rankings must be stable when the evaluation counter is swapped, and must match a blind human audit (about 100 images per model, Kendall τ ≥ 0.8). If they are unstable, return to Phase 1.

---

## Phase 3: Method (Task B SOTA, about 6–10 weeks)

Work through the options cheapest first, and stop when Gate 3 is met.

### 3a. Training-free stack

Use an LLM-generated layout, then a layout-grounded modern base model, then detector-driven self-correction (in the style of SLD), then best-of-N with the reward counter.

LMD+ plus SLD reaches 98% on *single-class* counts of 1–5. The open question is how it holds at 3 types and up to 15 instances. This arm is a strong baseline and possibly the result.

### 3b. Online RL (Flow-GRPO family)

- **Base:** SD3.5-M first, because the `yifan123/flow_grpo` code supports it and its 0.50 → 0.95 GenEval counting gain is the reference point. Move to Qwen-Image or FLUX if 3b-SD3.5 works and compute allows.
- **Reward:** the mean over types of `exp(−|ĉ−c*|/c*)` (ABACUS-style), minus λ × the extra-class indicator. The reward counter must differ from the evaluation counter.
- **Prompts:** the IOQM v3 train split. Test-OOD classes and count tuples are never seen in training.
- **Guarding against reward hacking:**
  - KL (β ≈ 0.04, as in Flow-GRPO)
  - a quality term (aesthetic or DeQA)
  - tracking GenEval overall, an aesthetic score, and a human "is this a normal photo" check at every checkpoint, to catch the documented "flat drawing" collapse
- **Compute:** check the Flow-GRPO repo for its reported GPU-hours before committing. I don't have a verified figure.

### 3c. Optional combination

RL-tuned model plus 3a self-correction plus best-of-N.

**Gate 3, the SOTA claim.** The method must meet all of the following:
- beat the best Phase-2 baseline on prompt-level exact match on **test-OOD**, with non-overlapping paired-bootstrap CIs
- be confirmed by blind human evaluation
- improve or match on at least 2 external benchmarks
- show no significant regression in image quality or GenEval overall

---

## Phase 4: Release

- IOQM v3 prompts and splits, plus IOQM-Human, on Hugging Face.
- An evaluation harness with the frozen counter and calibrated thresholds.
- Model weights or LoRA.
- A paper. Possible framing: *"Joint multi-class counting in T2I: a validated benchmark, a counter for generated images, and RL with held-out compositional evaluation."*

---

## Risks

| Risk | Mitigation |
|---|---|
| No counter reaches human agreement on fused objects | 1c fine-tuning. Report results on non-ambiguous images and on all images |
| Reward hacking or quality collapse in RL | Separate reward and evaluation counters, KL, quality term, human audit at each checkpoint |
| Scooped: M3T2IBench and NumBench are close | Differentiate on exhaustive enumeration, the co-occurrence and OOD split, 3-type exact match, and human-validated scoring. Move quickly on Phases 0–2 |
| Compute | Run the 3a training-free stack first. Start RL on SD3.5-M with LoRA |
| GenEval-style numbers drift across papers | Re-run every baseline in our own harness. Never copy numbers from other papers into a results table |

## Decisions (2026-11-01) and what they change

1. **Headline: the generator.** The counter is the measuring instrument. Phase 1c (fine-tuning a counter) runs only if Gate 1 fails, and is not a goal in itself.
2. **Compute: one A100 on Colab Pro, with hundreds of hours of credits.** This changes the plan in four ways:
   - **Check which A100 it is first.** Run `nvidia-smi` and note whether it has 40GB or 80GB. Colab's A100 is commonly 40GB.
     - On 40GB, Qwen-Image (~20B transformer plus a 7B text encoder) and FLUX.1-dev (12B) need CPU offload or fp8/8-bit weights for inference. That makes them slower, so measure seconds per image in Phase 0.
   - **Colab sessions get disconnected.** Every generation and training job must be resumable:
     - skip images that already exist (keyed by prompt_id and seed)
     - checkpoint RL every N steps to Google Drive
     - write results as append-only JSONL
   - **RL is limited to SD3.5-M with LoRA on a single GPU.** Flow-GRPO's published setup is multi-GPU. Before committing, run a **feasibility pilot** (~10 GPU-hours) that measures:
     - steps per hour at a reduced group size and batch size
     - whether the reward moves at all within that budget

     RL on Qwen-Image or FLUX at full scale is out of reach. A training-free stack on those bases is not.
   - **Revised compute budget (estimate):**
     - Phase 0 and 1: ~20h
     - Phase 2 baselines on the ~3k-prompt × 4-seed subset, 8 open models: ~100–150h. Trim to 2 seeds or fewer models if per-image times come in high.
     - Phase 3a training-free stack: ~30h
     - Phase 3b RL pilot plus a full run: ~100–200h
     - That totals several hundred hours, so it fits, but the budget is tight. The Phase 0 timing measurements decide the final scope.
3. **Annotation: the user labels the human set alone.** With a single annotator, inter-annotator agreement can't be measured. Replace it as follows:
   - **Intra-rater agreement as the human ceiling.** Re-label a blind random ~200 images at least 2 weeks after first labeling them, in random order and without seeing the earlier labels.
   - **Later, a paid second annotator** for ~300 images, alongside the API spend, to get a true inter-annotator figure for the paper. Reviewers will expect it.
   - **Upgrade the existing Gradio labeler** (`notebooks/human_ioqm_eval.ipynb`):
     - point clicks for each instance
     - an "ambiguous / fused" flag
     - the artifact-counting rule shown on screen
     - output saved as JSONL with image_id, generator and seed
   - **Labeling time (estimate):** ~30–60 s per image, so ~15–25 hours for 1,500 images.
4. **Closed-model APIs: yes, but later.** Phase 2 starts with open models only. The harness gets an API backend interface now, so GPT-Image, Gemini image and Seedream can be added for the paper's final table without rework.
