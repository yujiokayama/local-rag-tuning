"""実行環境の記録（記事に「どのマシンで測ったか」を書くため）"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import urllib.request
from importlib import metadata

PKGS = ["llama-index-core", "llama-index-llms-ollama", "llama-index-embeddings-ollama",
        "llama-index-vector-stores-chroma", "chromadb", "rank-bm25", "fugashi",
        "sentence-transformers", "torch"]


def _sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def http_json(url: str, timeout: float = 3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def hardware() -> dict:
    hw = {"os": platform.platform(), "arch": platform.machine(), "python": platform.python_version()}
    if platform.system() == "Darwin":
        hw["chip"] = _sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _sh(["sysctl", "-n", "hw.memsize"])
        hw["memory_gb"] = round(int(mem) / 1024**3) if mem else None
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            hw["memory_gb"] = round(kb / 1024**2)
        except Exception:
            pass
        hw["chip"] = _sh(["sh", "-c", "lscpu | grep 'Model name' | head -1 | cut -d: -f2"])
        hw["gpu"] = _sh(["sh", "-c", "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1"])
    return hw


def collect_env(cfg: dict, mock: bool = False) -> dict:
    hw = hardware()
    env = {
        "machine_label": os.environ.get("RAG_MACHINE") or f"{hw.get('chip') or hw['arch']} / {hw.get('memory_gb')}GB",
        "hardware": hw,
        "packages": {},
    }
    for p in PKGS:
        try:
            env["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    if not mock:
        base = cfg["ollama"]["base_url"]
        env["ollama_version"] = (http_json(f"{base}/api/version") or {}).get("version")
        tags = (http_json(f"{base}/api/tags") or {}).get("models", [])
        want = {cfg["ollama"]["llm"], cfg["ollama"]["embed"]}
        env["models"] = {
            m["name"]: {"digest": m.get("digest", "")[:12], "size_gb": round(m.get("size", 0) / 1024**3, 1),
                        "quant": (m.get("details") or {}).get("quantization_level")}
            for m in tags
            if m["name"] in want or m["name"].split(":")[0] in want or m["name"].removesuffix(":latest") in want
        }
        env["ollama_ps"] = http_json(f"{base}/api/ps")
    return env
