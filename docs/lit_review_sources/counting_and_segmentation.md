# Choosing a counter for IOQM: literature review (as of 2026-11-01)

Everything below was checked against arXiv abstract/HTML pages, the arXiv API (titles, dates, venue comments), GitHub READMEs, or the GenEval PDF text. Numbers come from the table named in each row. "LB" means the number is taken from the FSC-147 leaderboard at github.com/Mengqi-Lei/awesome-object-counting, not from the original paper.

## 1. Methods table

| Method | Venue / year | arXiv | Type | Open weights? | Key reported number (benchmark, metric, value, source) | Relevance to IOQM |
|---|---|---|---|---|---|---|
| FamNet + FSC-147 | CVPR 2021 | 2104.08391 | density, exemplar | yes | FSC-147 test MAE 22.08 / RMSE 99.54 (LB; HieraCount Tab. 3) | Historical baseline only |
| CounTR | BMVC 2022 | 2208.13721 | density, exemplar | yes | FSC-147 test 11.95 / 91.23 (LB) | Superseded |
| LOCA | ICCV 2023 | 2211.08217 | density, exemplar | yes | FSC-147 test 10.79 / 56.97 (LB) | Superseded |
| CounTX | BMVC 2023 | 2306.01851 | density, text | yes | FSC-147 test 15.73 / 106.88, text only (LB) | Superseded |
| CLIP-Count | ACM MM 2023 | 2305.07304 | density, text | yes | FSC-147 test 17.78 / 106.62 (LB) | Superseded |
| VLCounter | AAAI 2024 | 2312.16580 | density, text | yes | FSC-147 test 17.05 / 106.16 (LB) | Superseded |
| PseCo | CVPR 2024 | 2311.12386 | detection (SAM + CLIP) | yes | FSC-147 test 13.05 / 112.86 (LB, detection protocol) | Superseded |
| DAVE | CVPR 2024 | 2404.16622 | detect-and-verify | yes | FSC-147 test 8.66 / 32.36 (LB) | Exemplar-only, so a poor fit for text prompts |
| GeCo | NeurIPS 2024 | 2409.18686 | detection + segmentation, exemplar | yes | FSC-147 test 7.91 / 54.28 (LB) | Needs exemplar boxes |
| T-Rex2 | Tech report 2024 | 2403.14610 | detector, text + visual prompts | API | FSC-147 MAE 10.9 with visual prompts (reported in Rex-Omni Tab. 6) | Closed |
| CountGD | NeurIPS 2024 | 2407.04619 | detector-based counter (GDINO) | yes | FSC-147 test 6.75 / 43.65 (text + exemplars); 5.74 / 24.09 on corrected test; text only 14.76 / 120.42 (Tab. 1). CountBench MAE 0.86, text only (Tab. 2) | Strong text-only counter; returns boxes |
| CountGD++ | CVPR 2026 | 2512.23351 | detector-based counter, positive and negative prompts | yes (HF nikigoli/CountGDPlusPlus) | FSCD-147 test: exemplars only 6.43 / 32.73; text + exemplars 5.86 / 18.79; text + pseudo/synthetic exemplars 8.39 / 27.03 (Tab. A.3). PairTally MAE 46.41 to 35.27 with negative prompts (CVPR supp.) | **Negative prompts fit the multi-class setting** |
| CountSE | ICCV 2025 | (CVF only) | text-only counter | not checked | FSC-147 test 7.84 / 82.99 (LB) | Best text-only on the leaderboard |
| YOLO-Count | ICCV 2025 | 2508.00728 | differentiable regression counter | not stated | FSC-147 test 14.80 / 96.14 (Tab. 1) | Built for T2I guidance, not evaluation |
| CoDi | arXiv 2025 (TIP 2026 per LB) | 2512.20153 | diffusion counter, exemplar | not checked | Abstract claims "15% MAE" gain over prior SOTA, few-shot; no absolute numbers seen | Exemplar-based |
| HieraCount + KubriCount | arXiv 2026 | 2605.10887 | text + exemplar | not checked | FSC-147 test 10.29 / 99.13 (Tab. 3); PairTally MAE 36.27 | Synthetic training data |
| Grounding DINO | ECCV 2024 | 2303.05499 | open-vocab detector | yes | COCO F1@0.5 69.8, Swin-T, zero-shot (Rex-Omni Tab. 2) | Detection baseline |
| Grounding DINO 1.5 Pro | Tech report 2024 | 2405.10300 | open-vocab detector | API | COCO 54.3 AP; LVIS-minival 55.7 AP zero-shot (abstract) | API only, so not reproducible |
| DINO-X Pro | Tech report 2024 | 2411.14347 | open-vocab detector | API (DINO-X-API repo) | COCO 56.0 AP zero-shot (abstract); CountBench 82.9% / PixMo-Count 85.0% (SAM 3 Tab. 3) | API only |
| OWLv2 / OWL-ST | NeurIPS 2023 | 2306.09683 | open-vocab detector | yes | LVIS rare AP 31.2 to 44.6, L/14 (abstract) | Weaker detector baseline |
| Florence-2 | CVPR 2024 | 2311.06242 | VLM / detector | yes | No counting number verified | Low priority |
| APE | CVPR 2024 | 2312.02153 | open-vocab detection + segmentation | yes | No counting number verified | Low priority |
| Grounded-SAM | arXiv 2024 | 2401.14159 | GDINO + SAM pipeline | yes | No counting number verified | Superseded by SAM 3 |
| SAM 2 | ICLR 2025 | 2408.00714 | promptable segmentation (no text input) | yes | Not a counter | n/a |
| **SAM 3** | arXiv Nov 2025 | 2511.16719 | open-vocab concept segmentation | yes (gated HF; SAM 3.1 released 2026-03-27) | CountBench MAE 0.12 / Acc 93.8; PixMo-Count 0.21 / 86.2 (Tab. 3); SA-Co/Gold cgF1 54.1 vs human 72.8 (Tab. 1) | **Best verified score at counts in IOQM's range** |
| Rex-Omni | arXiv Oct 2025 | 2510.12798 | MLLM detector (Qwen2.5-VL-3B) | yes (HF, IDEA License) | COCO F1@0.5 72.0 (Tab. 2); Dense200 F1 78.4 (Tab. 4); FSC-147 MAE 7.0 with visual prompts (Tab. 6) | Point/box output, open |
| Mask2Former | CVPR 2022 | 2112.01527 | closed-set COCO segmentation | yes | COCO instance segmentation 50.1 AP (abstract); **GenEval's scorer** | GenEval-comparable baseline |
| Co-DETR (Co-DINO-Inst ViT-L) | ICCV 2023 | 2211.12860 | closed-set COCO detection + segmentation | yes (HF) | COCO test-dev box 66.0 / mask 57.1 (GitHub README) | Stronger drop-in for DETR/DETA/YOLO |
| Molmo / PixMo-Count | CVPR 2025 | 2409.17146 | VLM, pointing | yes | Molmo-72B CountBench 92.4% / PixMo-Count 88.8%, MAE 0.17 (SAM 3 Tab. 3) | Pointing gives auditable counts |
| Molmo2 | arXiv 2026 | 2601.10611 | VLM, pointing, video | yes | Video counting 35.5 vs Qwen3-VL 29.6 (abstract) | Image counting not verified |
| Qwen2.5-VL | arXiv 2025 | 2502.13923 | VLM | yes | COCO F1@0.5 64.0, 7B (Rex-Omni Tab. 2); Dense200 F1 1.1 | Collapses on dense scenes |
| Qwen3-VL | arXiv Nov 2025 | 2511.21631 | VLM, boxes + points | yes | Paper claims SOTA on CountBench; table values UNVERIFIED | Candidate VLM judge |
| Gemini 2.5 Pro | closed | n/a | VLM | no | CountBench 92.4% / PixMo-Count 78.2% (SAM 3 Tab. 3) | Closed |
| CountBench | ICCV 2023 | 2302.12066 | benchmark (540 images, counts 2–10) | yes | Saturated: aggregator lists Qwen3.6-27B 97.8%, Gemini 3 Pro 97.3% (benchlm.ai, vendor-reported) | Closest count range to IOQM |
| TallyQA | AAAI 2019 | 1810.12440 | benchmark | yes | SOTA not verified | Low priority |
| CountQA | arXiv 2025 | 2508.06585 | benchmark (1,528 dense, real images) | yes | Best of 15 MLLMs: 42.9% (abstract) | Dense; not IOQM's regime |
| UNICBench | CVPR 2026 | 2603.00595 | MLLM benchmark | yes | 45 MLLMs; numbers not verified | Low priority |
| OmniCount / OmniCount-191 | AAAI 2025 | 2403.05435 | multi-label counting | yes | Numbers not verified | Multi-class |
| ABC123 / MCAC | arXiv 2023 (v2 2024) | 2309.04820 | multi-class, exemplar-free | yes | Numbers not verified | Multi-class |
| REC-8K / GroundingREC | CVPR 2024 | CVF (Dai et al.) | referring-expression counting | yes | 8,011 images, 17,122 expressions | Attribute-level counting |
| PrACo | WACV 2025 | 2409.15953 | prompt-grounding test (mosaic, two classes) | yes | SOTA counters "misread the prompt" | **Directly tests IOQM's failure mode** |
| PrACo++ / MUCCA | arXiv May 2026 | 2605.02752 | multi-category real images | yes | 10 SOTA counters ground class names weakly | Multi-class |
| PairTally | arXiv 2025 | 2509.13939 | 681 images, 2 categories each | yes | CountGD++ ± negative prompts, above | Multi-class |
| MixCount | arXiv May 2026 | 2605.18063 | synthetic 3D data, 2–4 classes per scene | yes | Fine-tuning CountGD++ on +1k images: FSC-147 MAE 5.86 to 4.68; PairTally 35.33 to 28.87 (Tab. 2) | **Fine-tuning recipe** |
| GenEval | NeurIPS 2023 D&B | 2310.11513 | T2I evaluation (Mask2Former) | yes | Counting κ 0.37 to 0.65 at threshold 0.9; 5-fold CV 0.823 ± 0.013 (App. C) | **Template for validation** |
| GenEval 2 | arXiv Dec 2025 | 2512.16853 | T2I evaluation (Soft-TIFA, Qwen3-VL-8B) | yes | GenEval error up to 17.7% on current models; Soft-TIFA AUROC 94.5 | Evaluators drift as generators change |
| T2I-CompBench++ | arXiv v3 2025 | 2307.06350 | T2I evaluation (UniDet numeracy) | yes | Numeracy Kendall τ 0.4251 / Spearman ρ 0.5273, vs GPT-4V 0.3777 / 0.4651 (Tab. XII) | Detector beats VLM judge |
| GeckoNum | arXiv 2024 (v3 2025) | 2406.14774 | T2I numeracy + human ratings | yes | Only VNLI and Gecko separate correct from incorrect images across all 5 models (Tab. 3, Mann-Whitney) | QA-based raters unreliable |
| Make It Count / CoCoCount | CVPR 2025 | 2406.10210 | T2I; YOLOv9 vs MTurk | n/a | Aggregate YOLOv9 vs human: SDXL 28/26, DALL-E 3 25/38 (Tab. 1) | Detector bias varies by generator |
| NumBench | arXiv Aug 2026 | 2608.28206 | T2I numeracy; OW-DETR + GDINO + OWL-ViT ensemble | n/a | 14,400-image human study; Fleiss κ .82 at counts 1–25, .33 at 76–100 (Tab. 5) | Largest human validation found |

## 2. What was state of the art in late 2026, by sub-area

**Class-agnostic counting (FSC-147).** Exemplar-based: CountGD++ reaches 6.43 MAE with exemplars only (Tab. A.3). With text plus exemplars, CountGD (5.74 on the corrected test) and CountGD++ (5.86 MAE, 18.79 RMSE) are essentially tied. CountGD++ fine-tuned with MixCount reaches 4.68 / 15.66 (MixCount Tab. 2), the lowest verified number. Text-only: CountSE at 7.84 (leaderboard). FSC-147 is the wrong proxy for IOQM, though. Its RMSE is driven by high-count outliers; the "corrected" test differs by one image (7171) and still moves RMSE from 43.65 to 24.09. IOQM has counts of 1–5 and COCO classes, a regime where CountBench (counts 2–10) and PixMo-Count are much closer proxies.

**Open-vocabulary detection and segmentation.** SAM 3 (open weights, gated) has the best verified small-count numbers: CountBench MAE 0.12 / 93.8%, beating Molmo-72B, Gemini 2.5 Pro and DINO-X (Tab. 3). Rex-Omni is the best open MLLM-style detector on dense scenes (Dense200 F1 78.4). The strongest closed-set COCO detector with public weights is Co-DINO-Inst (66.0 box / 57.1 mask AP on test-dev). DINO-X and Grounding DINO 1.5/1.6 are API-only.

**VLM counting.** CountBench is saturated, with vendor-reported scores of 97–98% (Qwen3.6, Gemini 3 Pro). Dense, real-world counting is not solved: on CountQA the best of 15 MLLMs scored 42.9% at release. Qwen2.5-VL collapses on dense detection (Dense200 F1 1.1). For IOQM's 1–5 range, top VLMs are probably adequate, but the CountBench figures above are self-reported.

**Multi-class counting.** This is the axis that separates candidates, and it is the least mature. PrACo, PrACo++ and PairTally all show that counters trained for one class per image often count the wrong class. The only verified multi-class improvement is CountGD++ with negative prompts (PairTally MAE 46.41 to 35.27) plus MixCount fine-tuning (35.33 to 28.87). I found **no verified SAM 3 result on a benchmark with two classes in the same image**. SAM 3 is evaluated one prompt at a time, so this is an open question, not a known weakness.

**Counting on generated images.** The headline finding: **no paper I found validates a dedicated counter (CountGD, SAM 3 or a VLM pointer) against human counts on T2I images.** All existing human validations use detectors:
- GenEval: Mask2Former. Raising the threshold to 0.9 lifted counting κ from 0.37 to 0.65, and the paper reports that NMS did not help.
- Make It Count: YOLOv9 against MTurk raters. The gap is generator-dependent: DALL-E 3 scored 13 points lower under YOLOv9 than under humans.
- NumBench: a three-detector ensemble. κ is .82 at counts 1–25, and swapping in any single detector shifts the metric by at least .10.
- T2I-CompBench++: UniDet correlates with humans better than GPT-4V on numeracy (τ 0.425 vs 0.378).
- YOLO-Count: the authors counted their generated images by hand.

GenEval 2 shows that detector-based evaluators drift as generators improve (up to 17.7% absolute error).

**Failure modes (verified).**
- Detectors fire many low-confidence boxes on repeated same-class objects; GenEval fixes this with a threshold, not NMS.
- Recall drops for small and crowded objects (NumBench).
- Prompt or class confusion in multi-class images (PrACo, PrACo++).
- VLMs fail on dense scenes (Rex-Omni Tab. 4; CountQA).
- Fused or merged objects: NumBench treats these as ambiguous even for humans ("counted by visible bodies when separable").
- No paper quantifies detector behavior specifically on fused objects.

## 3. Recommended shortlist to benchmark on IOQM's ~600 human-labeled images

1. **SAM 3 (or SAM 3.1), one text prompt per class.** It has the best verified small-count accuracy (CountBench 93.8%), open weights, and masks that let you audit fused or duplicate instances. Sweep the score threshold per class with k-fold CV.
2. **CountGD++, text prompts plus negative prompts naming the other classes in the prompt.** It is the only counter with a verified multi-class gain, it has open weights, and it can be fine-tuned (MixCount shows fine-tuning works).
3. **Co-DINO-Inst ViT-L (closed-set COCO-80).** A stronger drop-in replacement for DETR-R50, DETA and YOLOv8x. It separates "better detector" from "open-vocabulary". Also run Mask2Former at GenEval's thresholds (0.3, and 0.9 for counting) for direct GenEval comparability.
4. **One open VLM judge (Qwen3-VL point-then-count, or Molmo pointing).** Points make the counts auditable. GenEval 2 uses Qwen3-VL-8B. Note that GeckoNum and T2I-CompBench++ both found VLM or QA raters weaker than detectors on numeracy.
5. *(Optional)* **Rex-Omni.** An open 3B model with strong dense-scene F1, and a useful extra perspective for the ensemble.

**Protocol.** Report per-class count accuracy, MAE and Cohen's κ against the human labels. Break results down by count (1–5) and by number of classes per prompt (1–3). Follow GenEval's Appendix C method: 5-fold CV over thresholds, reporting mean ± std. Also check ensemble agreement, following NumBench's evidence that ensembles are more stable.

**Fine-tuning.**
- **What is known to work:** fine-tuning CountGD++ on a small synthetic set (+1,000 MixCount images rendered in Blender with exact instance masks) cut MAE by 20% on FSC-147 and 18% on PairTally. Nothing I found fine-tunes a counter on T2I-generated images with human labels.
- **First step:** calibrate thresholds rather than retrain. That is what GenEval did, and it raised κ from 0.37 to 0.65.
- **Then fine-tune** CountGD++ or SAM 3 on human-annotated T2I images that include the artifacts (fused, partial, odd-scale objects) with an explicit labeling convention (e.g., NumBench's "visible bodies when separable"), mixed with MixCount-style multi-class synthetic data.
- **Data sizes:** ~600 labels is enough for calibration. Fine-tuning likely needs a few thousand box- or point-labeled generated images. That figure is my estimate, not a verified one.
- **Avoid overfitting:** hold out generators, given the generator-dependent gaps in Make It Count and the drift in GenEval 2.

## 4. Claims I could NOT verify

- The CountSE paper's own FSC-147 numbers (the PDF returned 403). A search summary claimed "2.99 test MAE"; only the leaderboard's 7.84 / 82.99 is used here.
- CountGD++ at 7.95 / 29.24 (HieraCount Tab. 3) conflicts with CountGD++'s own Tab. A.3 (5.86 / 18.79, text + exemplars). The protocols probably differ.
- "SAM3Count" text-only FSC-147 MAE 13.02: secondary source only (survey arXiv 2608.23845, Tab. VIII).
- Two SAM 3 numbers conflict: LVIS mask AP is 48.8 in the text and 48.5 in Table 1.
- GenEval's detector: the GenEval PDF says Mask2Former, while GenEval 2 calls it "MaskFormer from MMDetection." I go with the primary source.
- Qwen3-VL CountBench values: the paper's table did not render, so these are UNVERIFIED.
- Gemini 3 / GPT-5.x / Qwen3.6 CountBench and CountQA scores: aggregator and vendor-reported only.
- Grounding DINO 1.6 numbers; whether GeCo2 exists and its FSC-147 MAE; TallyQA SOTA; CoDi's absolute MAE.
- Open-weights status of CountSE, YOLO-Count, CoDi and HieraCount.
- Gemini 2.5/3 pointing-specific counting results.
- TIFA, DSG and VQAScore counting-specific human correlations. GeckoNum gives only significance-separation tests.
- Results for MCAC, OmniCount-191 and REC-8K: datasets exist, numbers not pulled.
- NumBench's per-detector human agreement: not reported in the paper.
- Any study of detector or counter behavior specifically on fused or melded objects: none found.
