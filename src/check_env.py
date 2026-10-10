#!/usr/bin/env python3
"""セットアップ確認: ライブラリ・Ollama・モデル・num_ctx の実効値をチェックする"""
from __future__ import annotations

import sys
import time

import common as C
from envinfo import collect_env, http_json

OK, NG, WARN = "✅", "❌", "⚠️ "


def main() -> int:
    cfg = C.load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    o = cfg["ollama"]
    bad = 0
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print(f"{NG} Python 3.10 以上が必要です"); bad += 1

    for mod in ["llama_index.core", "llama_index.llms.ollama", "llama_index.embeddings.ollama",
                "llama_index.vector_stores.chroma", "chromadb", "rank_bm25", "fugashi", "yaml"]:
        try:
            __import__(mod); print(f"{OK} import {mod}")
        except Exception as e:
            print(f"{NG} import {mod}: {e}"); bad += 1
    try:
        import sentence_transformers  # noqa: F401
        import torch
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{OK} sentence-transformers（リランカー用） device={dev}")
    except Exception:
        print(f"{WARN} sentence-transformers 未導入（リランク実験 30/42 だけ動きません）")

    try:
        C.tokenize("リモートワークは週3日まで", "fugashi"); print(f"{OK} fugashi 日本語トークナイズ")
    except Exception as e:
        print(f"{NG} fugashi: {e}"); bad += 1

    ver = http_json(f"{o['base_url']}/api/version")
    if not ver:
        print(f"{NG} Ollama に接続できません: {o['base_url']}（`ollama serve` または `brew services start ollama`）")
        return 1
    print(f"{OK} Ollama {ver.get('version')} @ {o['base_url']}")

    env = collect_env(cfg)
    for m in (o["llm"], o["embed"]):
        hit = [k for k in env.get("models", {}) if k == m or k.removesuffix(":latest") == m]
        if hit:
            print(f"{OK} model {m}  {env['models'][hit[0]]}")
        else:
            print(f"{NG} model {m} がありません → ollama pull {m}"); bad += 1
    if bad:
        return 1

    # 実際に1回ずつ呼んでみる
    t = time.perf_counter()
    vec = C.build_embed(cfg).get_text_embedding("動作確認")
    print(f"{OK} embedding dim={len(vec)} ({time.perf_counter() - t:.1f}s)")

    llm = C.build_llm(cfg)
    t = time.perf_counter()
    r = llm.complete("「OK」とだけ返答してください。")
    raw = r.raw
    ptok = raw.get("prompt_eval_count") if isinstance(raw, dict) else getattr(raw, "prompt_eval_count", None)
    print(f"{OK} LLM 応答: {C.strip_think(r.text)[:30]!r} ({time.perf_counter() - t:.1f}s, prompt_tokens={ptok})")
    print(f"   LlamaIndex が送る num_ctx = {C.effective_num_ctx(llm)}"
          f"（base.yaml の context_window={o['context_window']}。null なら LlamaIndex 既定値）")
    ps = http_json(f"{o['base_url']}/api/ps") or {}
    for m in ps.get("models", []):
        extra = {k: m[k] for k in ("context_length", "size_vram", "expires_at") if k in m}
        print(f"   ollama ps: {m.get('name')} {extra}")
    print(f"\n記録されるマシン名: {env['machine_label']}（環境変数 RAG_MACHINE で上書き可）")
    print("\nセットアップOK。次は `make mock` → `make run EXP=00_baseline SAMPLE=1` の順で確認してください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
