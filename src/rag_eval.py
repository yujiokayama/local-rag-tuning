#!/usr/bin/env python3
"""ローカルRAG 評価スクリプト

使い方:
  python src/rag_eval.py config/experiments/00_baseline.yaml
  python src/rag_eval.py 30_rerank            # experiments/ 配下は名前だけでも可
  python src/rag_eval.py 00_baseline --sample # 同梱サンプル文書・質問で動作確認
  python src/rag_eval.py 00_baseline --mock   # Ollama なしで配線だけ確認

出力:
  results/<実験名>/<日時>/results.jsonl  … 1問ごとの詳細（取得チャンク・回答・トークン数・時間）
  results/<実験名>/<日時>/summary.json   … 集計
  results/<実験名>/<日時>/config.yaml    … 実行時の設定スナップショット
  results/<実験名>/<日時>/env.json       … 実行環境（マシン・Ollama・モデル・ライブラリ）
  results/summary.csv                     … 全実行の集計を1行ずつ追記（compare.py が読む）
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import yaml

import common as C
from envinfo import collect_env


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace(" ", "").replace(",", "").lower()


def score_question(q: dict, final_nodes, answer: str, hit_k: int) -> dict:
    files = []
    for n in final_nodes:
        f = C.file_of(n.node)
        if f not in files:
            files.append(f)
    expected = q.get("expected_files", []) or []
    top_files = {C.file_of(n.node) for n in final_nodes[:hit_k]}
    found = [f for f in expected if f in top_files]
    rank = next((i + 1 for i, n in enumerate(final_nodes) if C.file_of(n.node) in expected), None)
    kws = q.get("expected_keywords", []) or []
    kw_hit = [k for k in kws if norm(k) in norm(answer)]
    # 根拠チャンク判定: 期待キーワードが「期待ファイル由来の」上位k件チャンクに入っているか。
    #   ファイル単位の hit は文書が少ないと上限に張り付くので、答えの書かれた箇所まで取れたかを見る。
    #   別文書の同じ数字（例: 「10日」が経費精算の「翌月10日」にも出る）で当たり扱いにしない。
    ev_text = norm("".join(n.node.get_content() for n in final_nodes[:hit_k] if C.file_of(n.node) in expected))
    ev_found = [k for k in kws if norm(k) in ev_text]
    top1 = C.file_of(final_nodes[0].node) if final_nodes else None
    return {
        "retrieved_files": files,
        "file_hit_all": bool(expected) and len(found) == len(expected),  # 期待ファイルが全部 top-k に入ったか
        "file_recall": (len(found) / len(expected)) if expected else None,
        "first_hit_rank": rank,
        "mrr": (1 / rank) if rank else 0.0,
        "hit_at1": bool(expected) and top1 in expected,
        "evidence_recall": (len(ev_found) / len(kws)) if kws else None,
        "evidence_hit_all": (len(ev_found) == len(kws)) if kws else None,
        "keyword_hit": f"{len(kw_hit)}/{len(kws)}",
        "correct": bool(kws) and len(kw_hit) == len(kws),
    }


def _f(x):
    return None if x is None else float(x)


def raw_get(raw, key):
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw.get(key)
    return getattr(raw, key, None)


def summarize(rows: list[dict]) -> dict:
    def agg(rs):
        if not rs:
            return {}
        n = len(rs)
        ev = [r for r in rs if r.get("evidence_hit_all") is not None]  # キーワード無しの問（q13等）は除外
        return {
            "n": n,
            "file_hit_all": round(sum(r["file_hit_all"] for r in rs) / n, 3),
            "file_recall": round(sum(r["file_recall"] or 0 for r in rs) / n, 3),
            "mrr": round(sum(r["mrr"] for r in rs) / n, 3),
            "hit_at1": round(sum(r["hit_at1"] for r in rs) / n, 3),
            "evidence_hit_all": round(sum(r["evidence_hit_all"] for r in ev) / len(ev), 3) if ev else None,
            "evidence_recall": round(sum(r["evidence_recall"] for r in ev) / len(ev), 3) if ev else None,
            "correct": round(sum(r["correct"] for r in rs) / n, 3),
            "avg_latency_s": round(sum(r["t_total"] for r in rs) / n, 2),
            "max_prompt_tokens": max((r["prompt_tokens"] or 0) for r in rs),
            "ctx_full_suspect": sum(r["ctx_full_suspect"] for r in rs),
            "gen_cut": sum(r["gen_cut"] for r in rs),
        }

    return {
        "all": agg(rows),
        "single": agg([r for r in rows if r.get("type") == "single"]),
        "multi": agg([r for r in rows if r.get("type") == "multi"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", default="00_baseline")
    ap.add_argument("--sample", action="store_true", help="同梱のサンプル文書・質問を使う")
    ap.add_argument("--mock", action="store_true", help="Ollama を使わず MockLLM/MockEmbedding で配線確認")
    ap.add_argument("--only", help="質問IDをカンマ区切りで指定して一部だけ実行")
    args = ap.parse_args()

    cfg = C.load_config(args.experiment, sample=args.sample or args.mock)
    questions = C.load_questions(cfg)
    if args.only:
        ids = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in ids]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = cfg["name"] + ("-mock" if args.mock else "") + ("-sample" if args.sample and not args.mock else "")
    out_dir = C.ROOT / "results" / tag / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"=== {cfg['name']}: {cfg.get('description', '')}")
    documents = C.load_documents(cfg)
    nodes = C.build_nodes(cfg, documents)
    print(f"[data] {len(documents)} docs → {len(nodes)} chunks / {len(questions)} questions")

    embed = C.build_embed(cfg, mock=args.mock)
    llm = C.build_llm(cfg, mock=args.mock)
    num_ctx = C.effective_num_ctx(llm)
    print(f"[llm] {cfg['ollama']['llm']}  num_ctx={num_ctx}  (config.context_window={cfg['ollama']['context_window']})")

    index, index_name = C.build_vector_index(cfg, nodes, embed, mock=args.mock)
    r = cfg["retrieval"]
    vec_k = r["top_k"] if r["mode"] == "vector" else r["hybrid"]["vector_top_k"]
    vec_retriever = index.as_retriever(similarity_top_k=vec_k)
    bm25 = C.BM25Search(nodes, r["hybrid"]["tokenizer"]) if r["mode"] == "hybrid" else None
    if args.mock and cfg["rerank"]["enabled"]:
        print("[mock] リランクは mock では無効化します（モデルDLを避けるため）")
        cfg["rerank"]["enabled"] = False
    reranker = C.build_reranker(cfg)
    template = C.load_prompt(cfg)
    hit_k = cfg["eval"]["hit_k"]

    watcher = None if args.mock else C.OllamaLogWatcher()
    if watcher is not None:
        print(f"[log] Ollama サーバログ: {watcher.path or '見つからず（OLLAMA_LOG で指定可。切り詰め検出はトークン数の目安のみ）'}")

    rows = []
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fout:
        for q in questions:
            t0 = time.perf_counter()
            if r["mode"] == "vector":
                cands = vec_retriever.retrieve(q["question"])
            elif r["mode"] == "hybrid":
                v = vec_retriever.retrieve(q["question"])
                b = bm25.search(q["question"], r["hybrid"]["bm25_top_k"])
                cands = C.rrf_fuse([v, b], k=r["top_k"], rrf_k=r["hybrid"]["rrf_k"])
            else:
                raise SystemExit(f"未知の retrieval.mode: {r['mode']}")
            t1 = time.perf_counter()
            final = reranker.postprocess_nodes(cands, query_str=q["question"]) if reranker else cands
            t2 = time.perf_counter()

            prompt = template.format(context=C.build_context(final), question=q["question"])
            error = None
            try:
                resp = llm.complete(prompt)
                answer = C.strip_think(resp.text)
                ptok = raw_get(resp.raw, "prompt_eval_count")
                ctok = raw_get(resp.raw, "eval_count")
                done_reason = raw_get(resp.raw, "done_reason")
            except Exception as e:  # タイムアウト等。1問の失敗で実験全体を止めない
                error = f"{type(e).__name__}: {e}"
                answer, ptok, ctok, done_reason = "", None, None, "error"
            t3 = time.perf_counter()
            truncated = watcher.truncations() if watcher else []
            gen_cut = done_reason in ("length", "error")

            row = {
                "id": q["id"],
                "type": q.get("type"),
                "question": q["question"],
                "expected_files": q.get("expected_files"),
                **score_question(q, final, answer, hit_k),
                "candidates": [
                    {"file": C.file_of(n.node), "node_id": n.node.node_id, "score": _f(n.score)} for n in cands
                ],
                "final": [
                    {"file": C.file_of(n.node), "node_id": n.node.node_id, "score": _f(n.score),
                     "text_head": n.node.get_content()[:80]} for n in final
                ],
                "answer": answer,
                "prompt_chars": len(prompt),
                "prompt_tokens": ptok,
                "completion_tokens": ctok,
                "done_reason": done_reason,
                "gen_cut": gen_cut,  # 生成が上限で打ち切られた / エラーになった
                "error": error,
                # Ollama は num_ctx を超えた入力を黙って切り詰める（実測では num_ctx の約半分まで縮められた）。
                # ログで検出できればそれを正とし、ログが無い時は num_ctx の45%超えを「疑い」とする
                "truncated_log": truncated[-1] if truncated else None,
                "ctx_full_suspect": bool(truncated) or bool(ptok and num_ctx and ptok >= num_ctx * 0.45),
                "t_retrieve": round(t1 - t0, 3),
                "t_rerank": round(t2 - t1, 3),
                "t_llm": round(t3 - t2, 3),
                "t_total": round(t3 - t0, 3),
            }
            rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            mark = "OK" if row["correct"] else "NG"
            warn = "  ⚠ctx満杯の疑い" if row["ctx_full_suspect"] else ""
            if gen_cut:
                warn += f"  ⚠生成打ち切り({error or 'num_predict 到達'})"
            ev = "-" if row["evidence_hit_all"] is None else int(row["evidence_hit_all"])
            print(f"  {q['id']:>5} [{q.get('type', '-'):6}] {mark} hit_all={int(row['file_hit_all'])} "
                  f"rank={row['first_hit_rank']} ev={ev} "
                  f"files={row['retrieved_files']} ptok={ptok} {row['t_total']:.1f}s{warn}")

    summary = summarize(rows)
    env = collect_env(cfg, mock=args.mock)
    meta = {"experiment": cfg["name"], "description": cfg.get("description", ""), "timestamp": ts,
            "index": index_name, "num_ctx": num_ctx, "chunks": len(nodes), "hit_k": hit_k}
    (out_dir / "summary.json").write_text(json.dumps({**meta, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = C.ROOT / "results" / "summary.csv"
    fields = ["timestamp", "experiment", "mock", "sample", "machine", "num_ctx", "chunks", "hit_k", "group", "n",
              "file_hit_all", "file_recall", "mrr", "hit_at1", "evidence_hit_all", "evidence_recall", "correct", "avg_latency_s", "max_prompt_tokens",
              "ctx_full_suspect", "gen_cut", "result_dir"]
    if csv_path.exists():  # 列が増えた場合は既存行を新しい列構成で書き直す（古い行の新列は空欄）
        with csv_path.open(encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            old_fields, old_rows = rd.fieldnames, list(rd)
        if old_fields != fields:
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(old_rows)
    new = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for group, s in summary.items():
            if not s:
                continue
            w.writerow({"timestamp": ts, "experiment": cfg["name"], "mock": args.mock,
                        "sample": args.sample or args.mock, "machine": env.get("machine_label"),
                        "num_ctx": num_ctx, "chunks": len(nodes), "hit_k": hit_k, "group": group, **s,
                        "result_dir": str(out_dir.relative_to(C.ROOT))})

    print("\n--- 集計")
    for group, s in summary.items():
        if s:
            ev = "-" if s["evidence_hit_all"] is None else f"{s['evidence_hit_all']:.2f}"
            print(f"  {group:6} n={s['n']:2} hit_all={s['file_hit_all']:.2f} recall={s['file_recall']:.2f} "
                  f"mrr={s['mrr']:.2f} hit@1={s['hit_at1']:.2f} evidence={ev} "
                  f"correct={s['correct']:.2f} avg={s['avg_latency_s']}s "
                  f"max_ptok={s['max_prompt_tokens']} ctx満杯疑い={s['ctx_full_suspect']} 生成打ち切り={s['gen_cut']}")
    print(f"\n保存先: {out_dir.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
