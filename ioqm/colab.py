"""
Small helpers for the Colab notebooks. Nothing here is needed outside Colab.
"""

from __future__ import annotations

import os
import subprocess


def sh(cmd: str) -> None:
    """Run a shell command, stream its output, and raise if it fails, so that
    'Run all' stops at the first broken step instead of carrying on."""
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"command failed (exit {p.returncode}): {cmd}")


def hf_login() -> str:
    """Log in to Hugging Face from the Colab secret HF_TOKEN, or by prompting.
    Exports the token so scripts launched by sh() can use it. Returns the user name."""
    from huggingface_hub import get_token, login, whoami
    token = None
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception:
        pass
    if token:
        login(token=token)
    else:
        login()
        token = get_token()
    os.environ["HF_TOKEN"] = token
    return whoami()["name"]


def gpu_memory_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return 0.0


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 2**30
