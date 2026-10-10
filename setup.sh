#!/usr/bin/env bash
# ローカルRAG チューニング実験のセットアップ（macOS / Apple Silicon 想定、Linux は Ollama 導入済みなら可）
#
#   ./setup.sh                    # 全部入り
#   WITH_RERANK=0 ./setup.sh      # リランカー（torch）を入れない
#   LLM=gemma3:12b ./setup.sh     # 別モデルも pull したい時
#
# 何度実行しても安全（入っているものはスキップ）
set -euo pipefail
cd "$(dirname "$0")"

LLM="${LLM:-qwen3:14b}"
EMBED="${EMBED:-bge-m3}"
WITH_RERANK="${WITH_RERANK:-1}"
PREFETCH_RERANK="${PREFETCH_RERANK:-1}"
RERANK_MODEL="${RERANK_MODEL:-BAAI/bge-reranker-v2-m3}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

step() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
die()  { printf "\033[1;31m✗ %s\033[0m\n" "$*" >&2; exit 1; }

OS="$(uname -s)"
step "環境: $OS $(uname -m)"

# ---- 1. Ollama -------------------------------------------------------------
step "Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  if [[ "$OS" == "Darwin" ]]; then
    command -v brew >/dev/null 2>&1 || die "Homebrew が必要です → https://brew.sh"
    brew install ollama
  else
    die "ollama が見つかりません。https://ollama.com/download の手順で導入してから再実行してください"
  fi
fi
ollama --version || true

if ! curl -sf "$OLLAMA_URL/api/version" >/dev/null; then
  if [[ "$OS" == "Darwin" ]] && command -v brew >/dev/null 2>&1 && brew list ollama >/dev/null 2>&1; then
    brew services start ollama
  else
    echo "Ollama サーバが起動していません。別ターミナルで 'ollama serve' を実行するか、Ollama.app を起動してください"
  fi
  for _ in $(seq 1 30); do curl -sf "$OLLAMA_URL/api/version" >/dev/null && break; sleep 1; done
  curl -sf "$OLLAMA_URL/api/version" >/dev/null || die "Ollama に接続できません: $OLLAMA_URL"
fi
echo "Ollama 起動中: $(curl -s "$OLLAMA_URL/api/version")"

# ---- 2. モデル --------------------------------------------------------------
step "モデル取得（初回は数GB〜十数GBのダウンロード）"
for m in "$LLM" "$EMBED"; do
  if ollama show "$m" >/dev/null 2>&1; then echo "  済: $m"; else ollama pull "$m"; fi
done

# ---- 3. Python --------------------------------------------------------------
step "Python 仮想環境 (.venv)"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    PY="$c"; break
  fi
done
if [[ -z "$PY" ]]; then
  [[ "$OS" == "Darwin" ]] && command -v brew >/dev/null 2>&1 || die "Python 3.10 以上が必要です"
  brew install python@3.12; PY=python3.12
fi
echo "  使用: $PY ($($PY --version))"
[[ -d .venv ]] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
if [[ "$WITH_RERANK" == "1" ]]; then
  step "リランカー用ライブラリ（torch を含むので少し時間がかかります）"
  python -m pip install -q -r requirements-rerank.txt
  if [[ "$PREFETCH_RERANK" == "1" ]]; then
    step "リランカーモデルを事前ダウンロード: ${RERANK_MODEL}（以降はオフラインで動きます）"
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('$RERANK_MODEL')"
  fi
fi

# ---- 4. 確認 ----------------------------------------------------------------
step "動作確認"
python src/check_env.py
