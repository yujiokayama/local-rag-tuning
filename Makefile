# 使い方は README.md を参照
PY      := .venv/bin/python
EXP     ?= 00_baseline
SAMPLE  ?=
FLAGS   := $(if $(SAMPLE),--sample,)

.PHONY: setup check mock chunks run all compare clean-index

setup:          ## 依存の導入・モデル取得・確認まで
	./setup.sh

check:          ## 環境チェック
	$(PY) src/check_env.py

mock:           ## Ollama なしで配線確認（サンプル文書・Mock モデル）
	$(PY) src/rag_eval.py $(EXP) --mock

chunks:         ## チャンク分割を書き出して目視確認  例: make chunks EXP=22_chunk512_sentence
	$(PY) src/dump_chunks.py $(EXP) $(FLAGS)

run:            ## 1実験を実行  例: make run EXP=30_rerank
	$(PY) src/rag_eval.py $(EXP) $(FLAGS)

all:            ## 全実験を順に実行
	@for f in config/experiments/*.yaml; do $(PY) src/rag_eval.py $$f $(FLAGS) || exit 1; done

compare:        ## 結果を Markdown 表に
	$(PY) src/compare.py

clean-index:    ## 埋め込みキャッシュ（ChromaDB）を削除
	rm -rf .chroma .chroma-mock
