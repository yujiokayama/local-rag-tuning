#!/usr/bin/env python3
"""results/summary.csv から実験ごとの最新結果を Markdown 表にする（Qiita に貼る用）

  python src/compare.py            # 実測のみ（mock/sample除外）
  python src/compare.py --all      # mock/sample も含める
"""
from __future__ import annotations

import argparse
import csv

import common as C

COLS = [("mrr", "MRR"), ("hit_at1", "hit@1"), ("evidence_hit_all", "根拠hit"), ("file_hit_all", "hit_all"),
        ("file_recall", "recall"), ("correct", "正答率"),
        ("avg_latency_s", "平均秒"), ("max_prompt_tokens", "最大ptok"), ("ctx_full_suspect", "ctx満杯疑い"), ("gen_cut", "生成打ち切り")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    path = C.ROOT / "results" / "summary.csv"
    if not path.exists():
        raise SystemExit("results/summary.csv がまだありません")
    latest: dict[tuple, dict] = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if not args.all and (r["mock"] == "True" or r["sample"] == "True"):
            continue
        key = (r["experiment"], r["group"])
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r
    for group in ("all", "single", "multi"):
        rows = sorted((v for (e, g), v in latest.items() if g == group), key=lambda r: r["experiment"])
        if not rows:
            continue
        print(f"\n### {group}（n={rows[0]['n']}）\n")
        print("| 実験 | num_ctx | chunks | k | " + " | ".join(h for _, h in COLS) + " |")
        print("|---|---:|---:|---:|" + "---:|" * len(COLS))
        for r in rows:
            print(f"| {r['experiment']} | {r['num_ctx']} | {r['chunks']} | {r['hit_k']} | "
                  + " | ".join(r.get(c) or "-" for c, _ in COLS) + " |")
    machines = {v["machine"] for v in latest.values()}
    print(f"\n計測マシン: {', '.join(sorted(machines))}")


if __name__ == "__main__":
    main()
