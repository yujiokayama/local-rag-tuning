#!/usr/bin/env python3
"""チャンク分割の結果を書き出して目視確認する（日本語の文の途中で切れていないか等）

  python src/dump_chunks.py 21_chunk256 [--sample]
  → results/chunks/<実験名>.md
"""
from __future__ import annotations

import argparse
import statistics

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", default="00_baseline")
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config(args.experiment, sample=args.sample)
    nodes = C.build_nodes(cfg, C.load_documents(cfg))
    out = C.ROOT / "results" / "chunks" / f"{cfg['name']}{'-sample' if args.sample else ''}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lens = [len(n.get_content()) for n in nodes]
    bad_end = [n for n in nodes if n.get_content().rstrip()[-1:] not in "。．.！？!?」）)】\n#|"]
    c = cfg["chunk"]
    lines = [f"# {cfg['name']}  splitter={c['splitter']} chunk_size={c['chunk_size']} overlap={c['chunk_overlap']}",
             "", f"- chunks: {len(nodes)}", f"- 文字数: min={min(lens)} median={int(statistics.median(lens))} max={max(lens)}",
             f"- 文末が句点等で終わっていないチャンク: {len(bad_end)}", ""]
    for i, n in enumerate(nodes):
        t = n.get_content()
        lines += [f"## [{i}] {C.file_of(n)}  ({len(t)}字)", "", "```text", t, "```", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"→ {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
