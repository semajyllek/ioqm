"""
Generation that resumes after a disconnect, with per-image metadata.

Layout of a run directory:
    <out_dir>/
        images/<prompt_id>_s<seed>.png
        manifest.jsonl      one line per finished image, appended and fsynced
        run_config.json     model config, prompt template, package versions

Resuming: an image is "done" when its manifest line exists AND its file
exists. A Colab disconnect can lose at most the image being generated when it
happened. Images are written to a temp file and then renamed, so a
half-written PNG is never treated as done.

torch, diffusers and PIL are imported lazily, so this module (and its tests)
can be imported without a GPU stack installed.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

from .prompts import PromptSpec, render


# ----------------------------------------------------------------- registry

@dataclass(frozen=True)
class ModelConfig:
    key: str
    hf_id: str
    kind: str = "diffusers"           # "diffusers" | "sdxl_ensemble" | "api"
    dtype: str = "bfloat16"
    call_kwargs: Dict[str, Any] = field(default_factory=dict)
    gated: bool = False               # needs `huggingface-cli login` + license acceptance
    params_b: Optional[float] = None  # approximate parameter count of the denoiser, in billions
    settings_source: str = ""         # where call_kwargs came from
    notes: str = ""


# Sampler settings come from each model card's diffusers example, checked
# 2026-11-01. Gated cards could not be read without a login. Their settings
# are marked "VERIFY", come from memory of the card, and must be checked
# before Phase 2.
REGISTRY: Dict[str, ModelConfig] = {m.key: m for m in [
    # --- 2023 models, kept for continuity with the original IOQM results
    ModelConfig("sd15", "stable-diffusion-v1-5/stable-diffusion-v1-5", dtype="float16",
                call_kwargs=dict(num_inference_steps=50, guidance_scale=7.5),
                params_b=0.86, settings_source="diffusers defaults",
                notes="the original runwayml/ repo was removed from HF; this is the official mirror"),
    ModelConfig("sdxl_base_refiner", "stabilityai/stable-diffusion-xl-base-1.0", kind="sdxl_ensemble",
                dtype="float16", call_kwargs=dict(num_inference_steps=40, high_noise_frac=0.8),
                params_b=3.5, settings_source="2023 run_gen.py ensemble_pipe",
                notes="matches the 2023 'stable-diffusion-xl-refiner-1.0' results (base+refiner ensemble)"),
    ModelConfig("sdxl_base", "stabilityai/stable-diffusion-xl-base-1.0", dtype="float16",
                call_kwargs=dict(num_inference_steps=40, guidance_scale=5.0),
                params_b=2.6, settings_source="diffusers defaults"),
    # --- current open models
    ModelConfig("sd35_medium", "stabilityai/stable-diffusion-3.5-medium",
                call_kwargs=dict(num_inference_steps=40, guidance_scale=4.5),
                gated=True, params_b=2.5, settings_source="VERIFY: model card (gated)",
                notes="Flow-GRPO RL base (Phase 3b)"),
    ModelConfig("sd35_large", "stabilityai/stable-diffusion-3.5-large",
                call_kwargs=dict(num_inference_steps=28, guidance_scale=3.5),
                gated=True, params_b=8.1, settings_source="VERIFY: model card (gated)"),
    ModelConfig("flux1_dev", "black-forest-labs/FLUX.1-dev",
                call_kwargs=dict(num_inference_steps=50, guidance_scale=3.5, height=1024, width=1024),
                gated=True, params_b=12, settings_source="VERIFY: model card (gated)"),
    ModelConfig("flux2_klein_4b", "black-forest-labs/FLUX.2-klein-4B",
                call_kwargs=dict(num_inference_steps=4, guidance_scale=1.0, height=1024, width=1024),
                params_b=4, settings_source="model card diffusers example"),
    ModelConfig("flux2_klein_9b", "black-forest-labs/FLUX.2-klein-9B",
                call_kwargs=dict(num_inference_steps=4, guidance_scale=1.0, height=1024, width=1024),
                gated=True, params_b=9, settings_source="VERIFY: assumed same as klein-4B card",
                notes="GenEval 0.85 / counting 0.83 per ERNIE-Image Tbl 4"),
    ModelConfig("qwen_image", "Qwen/Qwen-Image",
                call_kwargs=dict(num_inference_steps=50, true_cfg_scale=4.0, negative_prompt=" ",
                                 height=1328, width=1328),
                params_b=20, settings_source="model card diffusers example (1:1 = 1328x1328)",
                notes="20B transformer + 7B text encoder: expect CPU offload on a 40GB A100"),
    ModelConfig("z_image_turbo", "Tongyi-MAI/Z-Image-Turbo",
                call_kwargs=dict(num_inference_steps=9, guidance_scale=0.0, height=1024, width=1024),
                params_b=6, settings_source="model card diffusers example"),
    ModelConfig("z_image", "Tongyi-MAI/Z-Image",
                call_kwargs=dict(num_inference_steps=50, guidance_scale=4.0, negative_prompt="",
                                 height=1024, width=1024),
                params_b=6, settings_source="model card diffusers example (card uses 1280x720; 1:1 here)"),
]}

# Not registered yet:
#   FLUX.2-dev (32B): too large for a single A100 without quantization.
#   HiDream-I1-Full: needs a separately loaded Llama-3.1-8B text encoder.
#   BAGEL: not a diffusers pipeline (its own bagel-mot code).
# Add these once a loader is written and timed.


# ----------------------------------------------------------------- backends

class Backend(Protocol):
    config: ModelConfig

    def generate(self, prompt: str, seed: int) -> Any:  # returns a PIL.Image-like object with .save()
        ...


def _torch_dtype(name: str):
    import torch
    return getattr(torch, name)


class DiffusersBackend:
    def __init__(self, config: ModelConfig, device: str = "cuda", offload: str = "auto"):
        """offload: "none" | "model" | "sequential" | "auto" (try full GPU, fall back to model offload on OOM)."""
        import torch
        from diffusers import DiffusionPipeline

        self.config = config
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(config.hf_id, torch_dtype=_torch_dtype(config.dtype))
        self.offload = offload
        if offload == "auto":
            try:
                self.pipe.to(device)
                self.offload = "none"
            except torch.cuda.OutOfMemoryError:
                self.pipe.to("cpu")
                torch.cuda.empty_cache()
                self.pipe.enable_model_cpu_offload()
                self.offload = "model"
        elif offload == "none":
            self.pipe.to(device)
        elif offload == "model":
            self.pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        else:
            raise ValueError(offload)
        self.pipe.set_progress_bar_config(disable=True)  # run_generation prints its own progress

    def generate(self, prompt: str, seed: int):
        import torch
        # A CPU generator gives the same initial noise on any GPU or offload setup.
        g = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(prompt=prompt, generator=g, **self.config.call_kwargs).images[0]


class SDXLEnsembleBackend:
    """SDXL base + refiner, as in the 2023 run_gen.py (no torch.compile, for portability).
    Always loads fully onto the GPU; `offload` is accepted for interface parity but ignored
    (both models together fit easily on an A100)."""

    def __init__(self, config: ModelConfig, device: str = "cuda", offload: str = "none"):
        import torch
        from diffusers import DiffusionPipeline

        self.config = config
        self.offload = offload
        dt = _torch_dtype(config.dtype)
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dt, variant="fp16", use_safetensors=True).to(device)
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae, torch_dtype=dt, variant="fp16", use_safetensors=True).to(device)
        for p in (self.base, self.refiner):
            p.set_progress_bar_config(disable=True)

    def generate(self, prompt: str, seed: int):
        import torch
        kw = self.config.call_kwargs
        g = torch.Generator("cpu").manual_seed(seed)
        latents = self.base(prompt=prompt, num_inference_steps=kw["num_inference_steps"],
                            denoising_end=kw["high_noise_frac"], output_type="latent", generator=g).images
        return self.refiner(prompt=prompt, num_inference_steps=kw["num_inference_steps"],
                            denoising_start=kw["high_noise_frac"], image=latents, generator=g).images[0]


class APIBackend:
    """
    Base class for paid closed-model APIs (GPT-Image, Gemini image, Seedream).
    Deferred by user decision (2026-11-01). Subclass this and implement
    `_call`. API "seeds" are usually not honored, so the seed is used as a
    sample index and recorded as such. Never put API keys in config or
    manifests. Read them from the environment inside `_call`.
    """

    def __init__(self, config: ModelConfig, **_):
        self.config = config
        self.offload = "n/a"

    def _call(self, prompt: str, sample_index: int):
        raise NotImplementedError("API backends are deferred; implement a subclass when budget is approved")

    def generate(self, prompt: str, seed: int):
        return self._call(prompt, seed)


def load_backend(config: ModelConfig, device: str = "cuda", offload: str = "auto") -> Backend:
    if config.kind == "diffusers":
        return DiffusersBackend(config, device, offload)
    if config.kind == "sdxl_ensemble":
        return SDXLEnsembleBackend(config, device)
    if config.kind == "api":
        raise NotImplementedError(f"no API backend implemented for {config.key}")
    raise ValueError(config.kind)


# ------------------------------------------------------------------- runner

def images_dir(run_dir: Path) -> Path:
    """<run_dir>/images, unless images_path.txt points elsewhere (used for imported
    legacy runs where a symlink is impossible, e.g. on Colab's Drive mount)."""
    run_dir = Path(run_dir)
    ptr = run_dir / "images_path.txt"
    return Path(ptr.read_text().strip()) if ptr.exists() else run_dir / "images"


def image_name(prompt_id: str, seed: int) -> str:
    return f"{prompt_id}_s{seed}.png"


def completed(out_dir: Path) -> Set[Tuple[str, int]]:
    """(prompt_id, seed) pairs whose manifest line AND image file both exist."""
    out_dir = Path(out_dir)
    man = out_dir / "manifest.jsonl"
    done: Set[Tuple[str, int]] = set()
    if not man.exists():
        return done
    with open(man) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line from a disconnect
            if (images_dir(out_dir) / r["image"]).exists():
                done.add((r["prompt_id"], int(r["seed"])))
    return done


def _versions() -> Dict[str, str]:
    v = {"python": platform.python_version()}
    for mod in ("torch", "diffusers", "transformers", "accelerate"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            v["gpu"] = torch.cuda.get_device_name(0)
            v["gpu_mem_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}"
    except Exception:
        pass
    return v


def run_generation(
    backend: Backend,
    specs: Sequence[PromptSpec],
    seeds: Sequence[int],
    out_dir: Path,
    template: str = "{objects}",
    joiner: str = " and ",
    numerals: str = "digits",
    max_images: Optional[int] = None,
    log_every: int = 25,
) -> int:
    """Generate every (spec, seed) pair not already done. Returns the number of new images."""
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    render_cfg = dict(template=template, joiner=joiner, numerals=numerals)

    cfg_path = out_dir / "run_config.json"
    run_cfg = {"model": asdict(backend.config), "render": render_cfg,
               "offload": getattr(backend, "offload", None), "versions": _versions()}
    if cfg_path.exists():
        prev = json.loads(cfg_path.read_text())
        if prev["model"] != run_cfg["model"] or prev["render"] != render_cfg:
            raise ValueError(f"{out_dir} was started with a different model/render config; use a new out_dir")
    else:
        cfg_path.write_text(json.dumps(run_cfg, indent=2))

    done = completed(out_dir)
    todo = [(s, seed) for s in specs for seed in seeds if (s.prompt_id, seed) not in done]
    if max_images is not None:
        todo = todo[:max_images]
    print(f"[ioqm] {backend.config.key}: {len(done)} done, {len(todo)} to generate -> {out_dir}")

    n_new, t_start = 0, time.time()
    with open(out_dir / "manifest.jsonl", "a") as man:
        for spec, seed in todo:
            prompt = render(spec, **render_cfg)
            t0 = time.time()
            img = backend.generate(prompt, seed)
            dt = time.time() - t0
            name = image_name(spec.prompt_id, seed)
            tmp = img_dir / (name + ".tmp.png")
            img.save(tmp)
            os.replace(tmp, img_dir / name)
            rec = {
                "image": name, "prompt_id": spec.prompt_id, "seed": seed, "prompt": prompt,
                "objects": [list(o) for o in spec.objects], "split": spec.split, "dataset": spec.dataset,
                "model": backend.config.key, "gen_seconds": round(dt, 3), "timestamp": time.time(),
            }
            man.write(json.dumps(rec) + "\n")
            man.flush()
            os.fsync(man.fileno())
            n_new += 1
            if log_every and n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"[ioqm] {n_new}/{len(todo)}  {rate:.2f}s/img  eta {rate * (len(todo) - n_new) / 3600:.2f}h")
    return n_new


def read_manifest(out_dir: Path) -> List[dict]:
    """Manifest rows, one per image (the last line wins: an image regenerated
    after its file went missing is listed twice)."""
    rows: Dict[str, dict] = {}
    with open(Path(out_dir) / "manifest.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.pop(r["image"], None)
            rows[r["image"]] = r
    return list(rows.values())
