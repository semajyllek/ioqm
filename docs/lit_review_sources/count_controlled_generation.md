# IOQM literature review: T2I counting benchmarks and count-faithful generation (verified 2026-11-01)

Every number below was read from the cited arXiv page, HTML, or PDF in this session unless it is marked otherwise. "n/r" means the source does not report it. "UNVERIFIED" means I could not confirm it.

## 1. Short answer to the key question

**Do other benchmarks test joint multi-class counts? Yes, at least six do. None does it the way IOQM does.**

These benchmarks put two or more object types with separate counts in a single prompt:
- **GeckoNum:** 2-additive and 3-additive types, 100 prompts each, counts 1–5.
- **T2I-CompBench++ numeracy:** about 40% of its 1,000 prompts are multi-object, counts 1–8.
- **HRS-Bench counting:** medium and hard levels use "N1 obj1 and N2 obj2", with LVIS classes.
- **GenEval 2:** an optional count on each of 1–3 objects, counts 2–7.
- **M3T2IBench:** a count for every category, COCO classes.
- **NumBench:** double-category prompts with independent counts, 1–100.

IOQM still differs from all of them in four ways:
- **Exhaustive coverage.** None of them enumerates the full class × count combination space. GeckoNum says it "sweeps over combinations" but samples 100 prompts per type. NumBench says its balance is "marginal, not a full crossing". M3T2IBench, HRS-Bench and GenEval 2 sample at random.
- **Exact-match metric.** Among the joint-count benchmarks, none scores "every type exactly right" with a detector. They use per-noun scoring (GeckoNum), partial credit of 1/(2n) (T2I-CompBench++), F1 (HRS-Bench), a soft count error (M3T2IBench), or VQA probabilities (GenEval 2). The one all-or-nothing scorer is GenEval++: 280 prompts, judged by GPT-4.1, with an unspecified Multi-Count design. IOQM's hard metric (the fraction of types with an exact count) is itself per-type. A prompt-level "all types exact" rate would set it further apart.
- **Unnatural co-occurrences.** None includes them on purpose. HRS-Bench even has GPT-3.5 rewrite each prompt into a plausible scene.
- **The combination of features.** Counting on its own, or count 1, does not set IOQM apart. T2I-CompBench++ numeracy is a standalone numeracy category with joint counts 1–8. GeckoNum numeric-simple (1–10) and LMD numeracy (1–5) also test counting alone. What no other benchmark offers is all of these together: exhaustive enumeration, joint counts, COCO classes, exact-count scoring, and an automatic detector scorer. T2I-CompBench++ numeracy is closest in prompt meaning. M3T2IBench is closest in classes and scorer.

IOQM is weaker in one place. Most of these benchmarks checked their automatic scorer against humans:
- GenEval: 83% agreement against an 88% human ceiling. On counting, Cohen's κ rose from 0.37 to 0.65 once the confidence threshold was raised to 0.9.
- HRS-Bench: 93% Pearson correlation for F1.
- NumBench: κ = .82 on counts 1–25.
- M3T2IBench: r = 0.67.

IOQM's YOLO-based scorer has no published human check. The detector-drift critiques (Echo-4o, GenEval 2) apply to it directly.

## 2. Benchmark comparison

| Benchmark | Venue/year | arXiv | Counting setup (classes; counts; joint multi-class?) | Scorer | Human validation | How IOQM differs |
|---|---|---|---|---|---|---|
| GenEval | NeurIPS 2023 D&B | 2310.11513 | `counting`: 80 prompts, "a photo of [N] [OBJECT]s", 1 COCO-80 class, N ∈ {2,3,4}. `two_object`: 99 prompts, presence only, no counts. **No joint counts** | Mask2Former (MMDetection). Confidence threshold 0.3, raised to 0.9 for counting only | 6,000 annotations on 1,200 images. 83% overall agreement vs 88% inter-annotator. Counting is 22 points above CLIPScore. Counting κ 0.37→0.65 at the 0.9 threshold (App. C) | Joint counts, counts 1 and 5, exhaustive, ~16k vs 80 prompts |
| GenEval++ | Inside Echo-4o, 2025 | 2508.09987 | 280 prompts, 7 types × 40. Types include Count, Color/Count, Pos/Count and "Multi-Count". Multi-Count design not described | GPT-4.1 checklist. A prompt passes only if every condition holds | not found | Sampled, small, VLM judge |
| GenEval 2 | arXiv Dec 2025 | 2512.16853 | 800 prompts at 3–10 "atoms". 40 objects (20 animate, 20 inanimate, not COCO). Optional count per object, 2–7, 1–3 objects. **Joint counts possible**, frequency not tabulated | Soft-TIFA with Qwen3-VL-8B (AM/GM of per-atom P(yes)) | 23,226 human annotations. Soft-TIFA_GM AUROC 94.5 vs VQAScore 92.4 | Sampled, mixed with attributes and relations, VLM judge, no count 1 |
| T2I-CompBench++ (numeracy) | TPAMI 2025 (journal version) | 2307.06350 (v3) | 1,000 prompts (700 train / 300 test). 30% single-, 30% two-, 40% multi-object. Counts 1–8. About 150 ChatGPT-sourced nouns. Template:natural ≈ 4:1. **Joint counts: yes** | UniDet. Each object earns 1/(2n) for being detected and 1/(2n) more for the exact count | Kendall τ 0.4251, Spearman ρ 0.5273 (Tbl XII) | Not exhaustive, not COCO-only, partial credit |
| GeckoNum | NeurIPS 2024 D&B | 2406.14774 | 1,386 prompts. 7 exact-number types: numeric-simple 1–10, 2-additive and 3-additive 1–5 (100 prompts each), 2-additive-color 1–8, and others. About 40 nouns (food, animals, nature, objects). **Joint counts: yes** | Humans: 5 annotators per image, mode count, scored per noun | Human-labeled by design (47,957 annotations) | Generation is fixed and human-scored, no automatic scorer. Exhaustiveness not stated. Best exact-number accuracy is DALL·E 3 at 45.2% |
| HRS-Bench (counting) | ICCV 2023 | 2304.05390 | Classes and counts sampled from LVIS, scenes written by GPT-3.5. Easy: 1 type. Medium/hard: "N1 obj1 and N2 obj2". **Joint counts: yes (2 types)**. Count ranges per level not stated | UniDet, precision/recall/F1 | 93% Pearson correlation of human F1 vs automatic F1 | LLM-paraphrased natural scenes, F1 rather than exact match |
| DrawBench | NeurIPS 2022 (Imagen) | 2205.11487 | About 200 prompts in 11 categories, one of them Counting. "19 counting prompts" comes from a secondary source | Human side-by-side ratings | Human by design | Tiny, no automatic scorer |
| TIFA (v1.0) | ICCV 2023 | 2303.11897 | About 4K prompts, ~25K QA pairs. "Counting" is one of 12 question categories | VQA-based faithfulness | Paper reports human correlation (not fetched) | Counting is incidental, prompts are natural captions |
| DSG (DSG-1k) | ICLR 2024 | 2310.18235 | 1,060 prompts with dependency-graph questions. Counts appear as attributes | VQA | Paper reports human study (not fetched) | Counting is incidental |
| GenAI-Bench / VQAScore | CVPRW 2024 (inferred from a CVF supplement filename) | 2406.13743 | 1,600 designer prompts. "Counting" is an advanced skill and includes attribute-object pairs | VQAScore, P("Yes") | Human ratings | Natural prompts, single holistic score |
| T2ICountBench | arXiv 2025, v2 Jan 2026 | 2503.06884 | One object type per prompt, counts 1–15, 6 listed categories, scenes and styles varied. **No joint counts** | Humans: 5 graders, best-of-4 pass | Fleiss κ 0.58 | Single class. No model averages above 50% (Imagen-3 43%) |
| CoCoCount | CVPR 2025 (inside Make It Count) | 2406.10210 | 200 prompts, COCO classes, counts 2–10, single class | YOLOv9 plus humans | Human columns reported | Single class |
| **M3T2IBench** | arXiv Oct 2025 | 2510.23020 | 10,000 prompts. COCO-80 reduced to 66 classes after colour filtering. **A joint count for each category**, at most 5 instances per prompt. Randomly sampled, adds colours and relations | Mask2Former (0.3 threshold, NMS IoU 0.9) plus CLIP colour classifier. Bias = Σ\|n_k−m_k\| | 500 samples, r = 0.6711 vs VQAScore 0.40 | **Closest to IOQM**: same classes and detector family. Differs by random sampling, a cap of ≤5 instances in total, attribute/relation confounds, and a soft metric |
| **NumBench** | arXiv Aug 2026 | 2608.28206 | 640K prompts, 1,600 categories, counts 1–100. 80K double-category core prompts with independent counts. Layout and appearance factors. Balanced marginally, not fully crossed | Ensemble of 3 detectors (OW-DETR, Grounding DINO, OWL-ViT) with count-dependent thresholds (cw-NPS) | 14,400 images. κ .82 (1–25), .58 (26–50), .33 (76–100) | Huge-count regime, open vocabulary. Not exhaustive at small counts. Max 2 categories |
| QUANT-Bench (QUOTA) | WACV 2026 | 2411.19534 | FSC classes across visual domains | not fetched | not fetched | Single class, domain shift |
| Zafar et al. benchmark | WACV 2026 | 2408.11721 | FSC-147, 147 classes. 25 examples per number per class (paper). "Counts 1–25, 3,674 samples" is from a search snippet only | Detector/counter | n/r | Single class |

## 3. GenEval leaderboard (overall and counting)

**Caveat:** every GenEval counting number below is single-class, counts 2–4, scored by Mask2Former. None of it transfers directly to joint IOQM prompts. The table numbers also drift:
- FLUX.1-dev counting is reported as 0.74, 0.75 and 0.79 in three papers.
- SANA-1.5 4.8B counting is 0.86 in Flow-GRPO and 0.72 in Reflect-DiT, a 14-point gap.
- GenEval 2's human study found Gemini 2.5 Flash Image at 75.4 by GenEval's own scorer and 93.1 by human judgment. It concludes that "GenEval is effectively saturated."

| Model | Open? | Overall | Counting | Rewriter/RL | Source |
|---|---|---|---|---|---|
| SD1.5 | open | 0.43 | 0.35 | – | Flow-GRPO Tbl 1 (2505.05470) |
| SDXL | open | 0.55 | 0.39 | – | BAGEL Tbl 5 (2505.14683) |
| SD3 Medium | open | 0.62 | 0.63 | – | Qwen-Image Tbl 4 (2508.02324) |
| SD3-Medium | open | 0.74 | 0.72 | – | BAGEL Tbl 5; OmniGen2 Tbl 11 (different eval settings) |
| SD3.5-M | open | 0.63 | 0.50 | – | Flow-GRPO Tbl 1 |
| **SD3.5-M + Flow-GRPO** | open | **0.95** | **0.95** | RL on GenEval-style prompts | Flow-GRPO Tbl 1 |
| SD3.5 Large | open | 0.71 | 0.73 | – | Qwen-Image Tbl 4 |
| FLUX.1-dev | open | 0.66 | 0.74 | – | Qwen-Image Tbl 4 |
| FLUX.1-dev | open | 0.82 | 0.75 | rewriter | BAGEL Tbl 5 |
| FLUX.1-dev | open | 0.67 | 0.79 | – | OmniGen2 Tbl 11 (2506.18871v4) |
| FLUX.2-klein-9B | open | 0.85 | 0.83 | – | ERNIE-Image Tbl 4 (2605.25347) |
| FLUX.2 [dev] | open | not found | not found | – | – |
| Lumina-Image 2.0 | open | 0.73 | 0.67 | – | Qwen-Image Tbl 4 |
| SANA-1.5 4.8B | open | 0.81 | 0.86 | – | Flow-GRPO Tbl 1 |
| SANA-1.5 4.8B | open | 0.76 | 0.72 | – | Reflect-DiT Tbl 1 (2503.12271). Best-of-2048: 0.80 / 0.77 |
| HiDream-I1-Full | open | 0.83 | 0.79 | – | Qwen-Image Tbl 4 |
| Janus-Pro-7B | open | 0.80 | 0.59 | – | Qwen-Image Tbl 4; BAGEL Tbl 5 |
| Show-o2 7B | open | 0.76 | 0.58 | – | Show-o2 Tbl 5 (2506.15564) |
| BAGEL | open | 0.82 | 0.81 | – | BAGEL Tbl 5 |
| BAGEL† | open | 0.88 | 0.84 | rewriter | BAGEL Tbl 5 |
| Echo-4o (BAGEL + synthetic data) | open | 0.89 | 0.83 | – | Echo-4o Tbl 1 (2508.09987) |
| BLIP3-o 8B | open | 0.84 | n/r | rewriter | BLIP3-o Tbl 2 (2505.09568) |
| UniWorld-V1† | open | 0.84 | 0.81 | rewriter | OmniGen2 Tbl 11 |
| OmniGen2 | open | 0.95 | 0.93 | – (self-reported in v4; not reproduced elsewhere) | OmniGen2 Tbl 11 |
| Z-Image | open | 0.84 | 0.78 | – | Z-Image Tbl 9 (2511.22699) |
| Z-Image-Turbo | open | 0.82 | 0.77 | – | Z-Image Tbl 9 |
| Qwen-Image | open | 0.87 | 0.89 | – | Qwen-Image Tbl 4 |
| **Qwen-Image-RL** | open weights for base; RL variant availability UNVERIFIED | **0.91** | **0.93** | RL | Qwen-Image Tbl 4 |
| ERNIE-Image (w/o PE) | open? (UNVERIFIED) | 0.89 | 0.78 | – | ERNIE-Image Tbl 4 |
| ERNIE-Image (w/ PE) | – | 0.87 | 0.82 | prompt enhancer | ERNIE-Image Tbl 4 |
| HunyuanImage 3.0 | open | not found (report uses GSB human evaluation) | not found | – | 2509.23951 |
| DALL·E 3 | closed | 0.67 | 0.47 | – | BAGEL Tbl 5 |
| GPT Image 1 [High] / GPT-4o | closed | 0.84 | 0.85 | – | Qwen-Image Tbl 4; Flow-GRPO Tbl 1 |
| Seedream 3.0 | closed | 0.84 | **0.91** | – | Qwen-Image Tbl 4 |
| Gemini 2.5 Flash Image ("Nano Banana") | closed | GenEval 75.4 by GenEval's scorer, 93.1 by human judgment | GenEval 2 count: 70.1 | – | GenEval 2 (2512.16853) §human study, Tbl 5 |
| Imagen 4, Nano Banana Pro/2, GPT-Image-1.5/2, Seedream 4.0/5.0 | closed | not found | not found | – | – |

Joint-count proxies:
- **GenEval 2 count column (Tbl 5):** SD3.5-L 52.2, FLUX.1-dev 55.6, Bagel+CoT 55.6, Gemini 2.5 Flash Image 70.1, Qwen-Image 70.3.
- **GenEval 2 by atomicity:** FLUX.1-dev scores 25% at atomicity 6 and 0% at every higher level.
- **M3T2IBench Tbl 2:** every model has average count bias above 1. FLUX.1-dev has the lowest at 1.27.
- **NumBench Tbl 4:** at counts 1–25, CountGen scores .692 cw-NPS on single-category prompts and Bounded Attention scores .641 on double-category prompts. The paper says commercial single-category scores are "below .75". I did not extract commercial double-category scores, so these are not proven overall bests.

## 4. Methods

| Method | Venue/year | arXiv | Type | Mechanism | Reported counting result (benchmark, number, table) | Code |
|---|---|---|---|---|---|---|
| Counting Guidance (Kang et al.) | WACV 2025 (Oral) | 2306.17567 | training-free | Gradient of a reference-free counting net on predicted noise. **Multi-class: per-object attention masks** | As a baseline: CoCoCount 21/22 (YOLO/human), Make It Count Tbl 1. Zafar Tbl 1 full benchmark: 9.47% on SD1.4 | not checked |
| Make It Count / CountGen | CVPR 2025 | 2406.10210 | training-free plus a trained layout-completion net | Counts instances from self-attention features. A trained model predicts the shape and location of missing objects | CoCoCount: SDXL 28/26 → **50/54** (YOLOv9/human). T2I-CompBench-Count (human) 29 → 48 (Tbl 1). **Single class only** (stated limitation) | project page; repo not confirmed |
| Detection-Driven Count Optimization (Zafar et al.) | WACV 2026 | 2408.11721 | per-token optimization (light training) | Optimizes a counting token. Detector-driven dynamic scale, no backprop through the detector. Token reused for new prompts | Up to 9 objects: SD-Turbo 26.9 → **45.61%** (DETR). CountGen gets 31 in the same table. Full FSC-147: 12.63 → 24 (Tbl 1) | not found |
| YOLO-Count | ICCV 2025 | 2508.00728 | training-free guidance with a trained counter | Differentiable cardinality-map counter (YOLO-World based) used as guidance on SDXL-Turbo | Large counts (25–100) shown only in a figure, no table numbers | UNVERIFIED |
| CountCluster | arXiv 2025 (under review) | 2508.10710 | training-free | Clusters cross-attention maps into k separated blobs | "+18.5 pp average count accuracy" (abstract). Multi-class not stated | n/r |
| ATHENA | arXiv Mar 2026 | 2603.19676 | training-free | Estimates counts mid-sampling and applies count-aware noise corrections | No numbers in abstract | n/r |
| Bounded Attention | arXiv 2403.16990 (venue not on arXiv page) | 2403.16990 | training-free | Bounds attention leakage between subjects | NumBench double-category, counts 1–25: **.641 cw-NPS**. Best among open/specialized systems. Commercial double-category scores not extracted | n/r |
| LMD / LMD+ | arXiv (v3 2024) | 2305.13655 | training-free (LLM layout plus guidance or GLIGEN) | LLM writes boxes, then layout-grounded diffusion | LMD numeracy: SD 39% → LMD 62% → **LMD+ 86%** (Tbl 1). Layout-only upper bound 97% (Tbl 2). **Single class, counts 1–5, 10 COCO classes. The task also includes negation prompts ("not generating specific objects")** | yes (project page) |
| SLD | arXiv 2311.16090 | 2311.16090 | training-free self-correction | LLM plus detector find errors, then latent edits | LMD numeracy: DALL·E 3 38 → 61%. LMD+ 82 → **98%** (Tbl 1). SD 38 → 55 → 60% over 1 and 2 rounds (Tbl 2) | n/r |
| GenArtist | NeurIPS 2024 (Spotlight) | 2407.05600 | training-free agent | MLLM plans, verifies and edits | not fetched | n/r |
| MIGC | CVPR 2024 | 2402.05408 | trained adapter | Per-instance shading with a box layout | COCO-MIG numbers not fetched | n/r |
| 3DIS | arXiv | 2410.12669 | trained (depth layout) | Depth-driven decoupled instance synthesis | not fetched | n/r |
| M3T2IBench R&E | arXiv 2025 | 2510.23020 | training-free | Revise prompt into a positive/negative pair, then CFG | SD3 AlignScore 53.3 → 55.1, bias 1.52 → 1.43 (Tbl 2) | n/r |
| Attend-and-Excite, Divide & Bind, BoxDiff, GLIGEN, InstanceDiffusion, RPG, Ranni, LayoutGPT | 2023–24 | IDs not re-verified this session | training-free / adapter | attention or layout control | no counting numbers fetched | – |
| DDPO | arXiv (no venue on page) | 2305.13301 | RL | PPO over the denoising chain | – | – |
| DRaFT | ICLR 2024 | 2309.17400 | reward backprop | Differentiable reward fine-tuning | – | – |
| ImageReward / ReFL | arXiv (no venue on page) | 2304.05977 | reward FT | Learned preference reward plus ReFL | – | – |
| AlignProp | arXiv | 2310.03739 | reward backprop | LoRA plus gradient checkpointing | – | yes |
| Diffusion-DPO | CVPR 2024 | 2311.12908 | preference FT | DPO on Pick-a-Pic, 851K pairs | – | – |
| **Flow-GRPO** | NeurIPS 2025 (per repo) | 2505.05470 | online RL | GRPO with ODE→SDE sampling. Rule-based detector reward that **gives partial credit for near-miss counts**. KL β = 0.04 | **SD3.5-M GenEval counting 0.50 → 0.95, overall 0.63 → 0.95, two-object 0.78 → 0.99 (Tbl 1)**. Training prompts come from the GenEval templates, deduplicated against test | yes (yifan123/flow_grpo) |
| DanceGRPO | arXiv 2025 | 2505.07818 | online RL | GRPO across diffusion and flow models, multiple rewards | GenEval figure only from a secondary source (UNVERIFIED) | project page |
| MixGRPO | arXiv 2025 (v7 2026) | 2507.21802 | online RL | Mixed ODE–SDE sliding window, ~50% less training time | n/r | n/r |
| GRPO-Guard | arXiv Oct 2025 | 2510.22319 | RL fix | Ratio normalization plus gradient reweighting against "implicit over-optimization" | qualitative | n/r |
| Qwen-Image-RL | tech report 2025 | 2508.02324 | RL | RL stage after SFT | GenEval counting 0.89 → **0.93**, overall 0.87 → 0.91 (Tbl 4) | base weights open |
| CompGen (C-GRPO) | CVPR 2026 | 2511.18378 | RL plus synthetic curriculum | Scene-graph curriculum (cardinality is a difficulty axis). Reward is LLaVA-1.6-13B VQA P(yes) | SD1.5 GenEval overall 42.08 → 53.88 (Tbl 1). Counting sub-score not visible | n/r |
| **ABACUS** | SIGGRAPH Asia 2026 / TOG (arXiv comment) | 2606.23835 | SFT plus GRPO | Unified 3B model (UniLIP). The understanding branch counts each candidate. Reward exp(−\|ĉ−c*\|/c*) + 0.1·aesthetic. Trained on ~1M Pexels images with T-Rex2 human-verified counts | CoCoCount **71/77** (YOLOv9/human) vs CountGen 50/54. GenEval counting with YOLOv9 re-scoring **94** (Tbl 3). Multi-class not addressed | project page only |
| Echo-4o | arXiv 2025 | 2508.09987 | SFT on synthetic data | 179K GPT-4o images, including template instructions varying count | BAGEL counting 0.81 → 0.83 only (Tbl 1) | yes (dataset) |
| Ma et al., inference-time scaling | arXiv Jan 2025 | 2501.09732 | search | Random/zero-order/path search over noises with verifiers | FLUX.1-dev T2I-CompBench numeracy 0.6167 → 0.6789 (ImageReward search, Tbl 1). Warns of "verifier hacking" | n/r |
| Reflect-DiT | arXiv Mar 2025 | 2503.12271 | inference-time reflection (light training) | Fine-tuned Qwen2.5-VL-3B critic plus in-context refinement | SANA-1.0 counting 0.58 → **0.80** at N=20. Best-of-20 reaches 0.73. SANA-1.5 best-of-2048 reaches 0.77 (Tbl 1) | n/r |

**Reward hacking evidence:**
- **Flow-GRPO itself.** Dropping KL on GenEval cuts quality: DeQA falls from 4.01 to 2.77 and ImageReward from 1.03 to 0.44 (Tbl 2). The authors note hacking "still occasionally occurs".
- **GDRO (2601.02036).** At GenEval reward 0.89, Flow-GRPO images become "almost flat drawings" with a plain background.
- **GRPO-Guard (2510.22319).** The proxy reward keeps rising while quality and alignment degrade.
- **Hong et al. (2601.03468; venue "CVPR 2026 Findings" inferred only from a CVF URL path).** Reward ensembles only partly mitigate hacking. They propose an artifact-reward regularizer.
- **Ma et al.** Searching against a single verifier overfits to that verifier's bias.

Mitigations seen in the literature:
- a KL penalty
- an ensemble of rewards
- an artifact or quality regularizer
- a held-out judge, e.g. GenEval 2's Soft-TIFA, or Co-GRPO's zero-shot protocol (2512.22288, not fetched beyond the search snippet)

## 5. Verdict: the most promising routes to SOTA on IOQM-style joint counting

**1. Online RL (Flow-GRPO family) on a strong open base such as Qwen-Image or SD3.5/FLUX, with a per-type count reward on IOQM-template prompts.**
- **Why it is promising:**
  - It shows the largest verified lift on counting: SD3.5-M went from 0.50 to 0.95 on GenEval counting.
  - Qwen-Image's own RL stage added 0.04 counting on top of the strongest open base.
  - ABACUS shows count-reward GRPO beating every training-free method on CoCoCount (77 vs 54 by human judgment).
  - IOQM's combinatorial generator is a ready prompt source. The per-type reward exp(−|ĉ−c*|/c*), summed over types, maps directly onto joint prompts.
- **Risks:**
  - The demonstrated hacking (flat drawings, artifacts).
  - IOQM's scorer is the same kind of detector as the reward. Train against one detector (for example Mask2Former or a VLM counter) and evaluate with a different one (YOLOv8x) plus a human audit sample.
  - Hold out class pairs and count tuples from training, not just prompts.
  - Keep KL or GRPO-Guard-style clipping in the loop.

**2. Layout-grounded generation plus detector-in-the-loop self-correction (LMD+/GLIGEN/MIGC/Bounded Attention, then SLD).**
- **Why it is promising:**
  - It is the only training-free family verified near ceiling on numeracy: LMD+ 86%, then 98% after SLD. Flow-GRPO and ABACUS also reach about 0.95, but both are trained.
  - An explicit box per instance handles unnatural co-occurrences, because the layout comes from the LLM rather than the image prior.
  - On NumBench double-category prompts, Bounded Attention scores .641 at counts 1–25. That is the best among the open and specialized systems in my extract. I did not extract commercial double-category scores.
- **Risks:**
  - The 98% is on **single-class**, counts 1–5, 10-COCO-class prompts. That happens to match IOQM v1's class count, but not its joint structure.
  - Up to 15 boxes over 3 types raises leakage and overlap. Base models here are older (SD1.x/SDXL), so image quality lags.

**3. Inference-time verifier-guided search or reflection on top of route 1 or 2.**
- **Why it is promising:**
  - Reflect-DiT went from 0.58 to 0.80 counting with 20 samples.
  - Best-of-N with a detector verifier is trivial to apply to any base and composes with RL.
- **Risks:**
  - Verifier hacking (Ma et al.).
  - Cost grows linearly with N.
  - If the verifier is IOQM's own detector, the reported score is optimistic by construction. Use a different detector or VLM as the selector.

A practical recipe that follows from the evidence:
1. Use Qwen-Image or FLUX as the base.
2. Run Flow-GRPO with a per-type exact-count reward, KL, and an artifact regularizer, on IOQM templates with held-out class combinations.
3. Optionally add best-of-N with a second detector.

None of the verified methods reports results on 3-type joint prompts. That gap is a real contribution IOQM can claim.

## 6. Claims I could NOT verify
- GenEval counting/overall for FLUX.2 [dev], HunyuanImage 3.0, Imagen 4, Nano Banana Pro/2, GPT-Image-1.5/2, Seedream 4.0/5.0. Not found in any primary table.
- OmniGen2's 0.95/0.93. Self-reported in v4 Table 11, with no independent reproduction. FLUX.1-dev counting in that same table (0.79) differs from other reports (0.74/0.75).
- GenEval 2 text says Gemini counting is 69.2. Table 5 says 70.1 (I used the table).
- Design of the GenEval++ "Multi-Count" task, and any human validation of GenEval++.
- Count ranges per HRS-Bench difficulty level. "3,000 prompts per skill" comes from a secondary listing.
- DrawBench "19 counting prompts" (secondary source).
- YOLO-Count's T2I numbers (figure only) and its code release.
- DanceGRPO's GenEval figure (0.522 from a secondary review).
- Venues of Bounded Attention, LMD, SLD, DDPO, ImageReward, 3DIS. The arXiv pages list none. My recollections (ECCV 2024, TMLR, CVPR 2024, ICLR 2024, NeurIPS 2023) were not confirmed.
- arXiv IDs for Attend-and-Excite, Divide & Bind, BoxDiff, GLIGEN, InstanceDiffusion, RPG, Ranni, LayoutGPT. Not re-verified this session.
- CompGen's GenEval counting sub-score before and after.
- Whether the Qwen-Image-RL weights are released.
- Whether any 2025–2026 paper evaluates on IOQM. A search for "IOQM" returned nothing.
- Zafar benchmark "counts 1–25, 3,674 samples" (search snippet only).
