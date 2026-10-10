"""ローカルRAG チューニング実験の共通部品（設定読み込み・分割・インデックス・検索・LLM）"""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = "data/docs.sample"
SAMPLE_QUESTIONS = "eval/questions.sample.yaml"


# ---------------------------------------------------------------- 設定

def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(exp_path: str | None, sample: bool = False) -> dict:
    base = yaml.safe_load((ROOT / "config/base.yaml").read_text(encoding="utf-8"))
    exp: dict = {}
    name = "base"
    if exp_path:
        p = Path(exp_path)
        if not p.exists():
            p = ROOT / "config/experiments" / (exp_path if exp_path.endswith(".yaml") else exp_path + ".yaml")
        exp = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        name = p.stem
    cfg = deep_merge(base, exp)
    cfg["name"] = name
    if sample:
        cfg["data"]["docs_dir"] = SAMPLE_DOCS
        cfg["data"]["questions"] = SAMPLE_QUESTIONS
    return cfg


def load_questions(cfg: dict) -> list[dict]:
    qs = yaml.safe_load((ROOT / cfg["data"]["questions"]).read_text(encoding="utf-8")) or []
    if not qs:
        raise SystemExit(
            f"質問が0件です: {cfg['data']['questions']}\n"
            "  前回の13問を書くか、動作確認だけなら --sample を付けてください。"
        )
    return qs


# ---------------------------------------------------------------- 文書と分割

def load_documents(cfg: dict):
    from llama_index.core import SimpleDirectoryReader

    docs_dir = ROOT / cfg["data"]["docs_dir"]
    files = [p for p in docs_dir.rglob("*") if p.is_file() and not p.name.startswith(".")]
    if not files:
        raise SystemExit(
            f"文書が見つかりません: {cfg['data']['docs_dir']}\n"
            "  前回の文書を置くか、動作確認だけなら --sample を付けてください。"
        )
    reader = SimpleDirectoryReader(input_files=[str(p) for p in sorted(files)], filename_as_id=True)
    docs = reader.load_data()
    for d in docs:
        apply_metadata_policy(d)
    return docs


# 埋め込みに渡すメタデータ / LLM に渡すメタデータ
#   既定のままだと file_path（PC固有の絶対パス）が埋め込み・プロンプトの両方に入り、
#   別PCで結果が変わる・検索にパス文字列のノイズが乗る。
#   埋め込みは本文のみ（BM25 と条件を揃える）、LLM には出典用に file_name だけ渡す。
EMBED_METADATA_KEYS: list[str] = []
LLM_METADATA_KEYS: list[str] = ["file_name"]


def apply_metadata_policy(doc) -> None:
    keys = list(doc.metadata.keys())
    doc.excluded_embed_metadata_keys = [k for k in keys if k not in EMBED_METADATA_KEYS]
    doc.excluded_llm_metadata_keys = [k for k in keys if k not in LLM_METADATA_KEYS]


JA_SECONDARY_REGEX = r"[^。！？!?\n]+[。！？!?\n]?|[。！？!?\n]"


def _rel_id(doc) -> str:
    """フォルダを移動・別PCにコピーしても同じIDになるよう、プロジェクトからの相対パスを使う"""
    fp = doc.metadata.get("file_path")
    if fp:
        try:
            return Path(fp).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return Path(fp).name
    return doc.doc_id


def build_splitter(cfg: dict):
    from llama_index.core.node_parser import SentenceSplitter

    c = cfg["chunk"]
    kwargs: dict[str, Any] = dict(
        chunk_size=c["chunk_size"],
        chunk_overlap=c["chunk_overlap"],
        # ノードIDを決定的にする（RRF融合やキャッシュ照合のため）
        id_func=lambda i, doc: f"{_rel_id(doc)}::{i}",
    )
    if c["splitter"] == "sentence_ja":
        # 段落は空行1つで区切り、文は日本語の句点・感嘆符・改行で区切る
        kwargs.update(
            paragraph_separator="\n\n",
            chunking_tokenizer_fn=lambda text: [s for s in re.findall(JA_SECONDARY_REGEX, text) if s],
            secondary_chunking_regex=JA_SECONDARY_REGEX,
        )
    elif c["splitter"] != "sentence":
        raise SystemExit(f"未知の splitter: {c['splitter']}")
    return SentenceSplitter(**kwargs)


def build_nodes(cfg: dict, documents):
    return build_splitter(cfg).get_nodes_from_documents(documents)


def file_of(node) -> str:
    return node.metadata.get("file_name") or Path(node.metadata.get("file_path", "?")).name


# ---------------------------------------------------------------- モデル

def build_llm(cfg: dict, mock: bool = False):
    if mock:
        from llama_index.core.llms import MockLLM
        return MockLLM(max_tokens=32)
    from llama_index.llms.ollama import Ollama

    o = cfg["ollama"]
    extra: dict[str, Any] = {}
    if o.get("seed") is not None:
        extra["seed"] = o["seed"]
    if o.get("num_predict"):
        extra["num_predict"] = o["num_predict"]
    kwargs: dict[str, Any] = dict(
        model=o["llm"],
        base_url=o["base_url"],
        temperature=o["temperature"],
        request_timeout=o["request_timeout"],
        keep_alive=o.get("keep_alive", "5m"),
        additional_kwargs=extra,
    )
    if o.get("context_window"):
        kwargs["context_window"] = o["context_window"]
    if o.get("thinking") is not None:
        kwargs["thinking"] = o["thinking"]
    return Ollama(**kwargs)


def effective_num_ctx(llm) -> int | None:
    """LlamaIndex が実際に Ollama へ送る num_ctx
    （llama-index-llms-ollama 0.11.0 では未指定時にモデルの最大長を /api/show で取得して送る）"""
    fn = getattr(llm, "get_context_window", None)
    return fn() if fn else None


def build_embed(cfg: dict, mock: bool = False):
    if mock:
        from llama_index.core.embeddings import MockEmbedding
        return MockEmbedding(embed_dim=1024)
    from llama_index.embeddings.ollama import OllamaEmbedding

    return OllamaEmbedding(model_name=cfg["ollama"]["embed"], base_url=cfg["ollama"]["base_url"])


# ---------------------------------------------------------------- ベクトルインデックス（ChromaDB 永続化）

def index_key(cfg: dict, nodes, mock: bool) -> str:
    """埋め込みモデル・分割設定・埋め込む文字列が1つでも変わればキーが変わる → 古いキャッシュを誤用しない
    （メタデータ込みの「実際に埋め込む文字列」でハッシュするので、メタデータ方針を変えても別インデックスになる）"""
    from llama_index.core.schema import MetadataMode

    h = hashlib.sha1()
    h.update(("mock" if mock else cfg["ollama"]["embed"]).encode())
    h.update(repr(sorted(cfg["chunk"].items())).encode())
    for n in nodes:
        h.update(n.node_id.encode())
        h.update(hashlib.sha1(n.get_content(metadata_mode=MetadataMode.EMBED).encode()).digest())
    return "rag_" + h.hexdigest()[:16]


def build_vector_index(cfg: dict, nodes, embed_model, mock: bool = False, log=print):
    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    path = ROOT / (".chroma-mock" if mock else ".chroma")
    client = chromadb.PersistentClient(path=str(path))
    name = index_key(cfg, nodes, mock)
    col = client.get_or_create_collection(name)
    store = ChromaVectorStore(chroma_collection=col)
    if col.count() == len(nodes):
        log(f"[index] キャッシュを再利用: {name}（{len(nodes)} chunks）")
        return VectorStoreIndex.from_vector_store(store, embed_model=embed_model), name
    if col.count():
        client.delete_collection(name)
        col = client.get_or_create_collection(name)
        store = ChromaVectorStore(chroma_collection=col)
    log(f"[index] 埋め込みを作成: {name}（{len(nodes)} chunks）")
    ctx = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex(nodes, storage_context=ctx, embed_model=embed_model, show_progress=True)
    return index, name


# ---------------------------------------------------------------- BM25 / ハイブリッド

_tagger = None


def tokenize(text: str, kind: str) -> list[str]:
    text = text.lower()
    if kind == "whitespace":
        return text.split()
    if kind == "bigram":
        s = re.sub(r"\s+", "", text)
        return [s[i:i + 2] for i in range(max(len(s) - 1, 1))]
    if kind == "fugashi":
        global _tagger
        if _tagger is None:
            import fugashi
            _tagger = fugashi.Tagger()
        out = []
        for w in _tagger(text):
            pos = w.feature.pos1 if hasattr(w.feature, "pos1") else ""
            if pos in ("補助記号", "空白") or not w.surface.strip():
                continue
            out.append(w.surface)
        return out
    raise SystemExit(f"未知の tokenizer: {kind}")


class BM25Search:
    def __init__(self, nodes, tokenizer: str):
        from rank_bm25 import BM25Okapi

        self.nodes = nodes
        self.kind = tokenizer
        self.bm25 = BM25Okapi([tokenize(n.get_content(), tokenizer) for n in nodes])

    def search(self, query: str, k: int):
        from llama_index.core.schema import NodeWithScore

        scores = self.bm25.get_scores(tokenize(query, self.kind))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [NodeWithScore(node=self.nodes[i], score=float(scores[i])) for i in order]


def rrf_fuse(result_lists, k: int, rrf_k: int = 60):
    """Reciprocal Rank Fusion：順位だけを使って複数の検索結果を混ぜる"""
    from llama_index.core.schema import NodeWithScore

    score: dict[str, float] = {}
    node_by_id = {}
    for results in result_lists:
        for rank, nws in enumerate(results, start=1):
            nid = nws.node.node_id
            node_by_id[nid] = nws.node
            score[nid] = score.get(nid, 0.0) + 1.0 / (rrf_k + rank)
    best = sorted(score, key=score.get, reverse=True)[:k]
    return [NodeWithScore(node=node_by_id[i], score=score[i]) for i in best]


# ---------------------------------------------------------------- リランカー

def build_reranker(cfg: dict):
    if not cfg["rerank"]["enabled"]:
        return None
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"リランカーの読み込みに失敗: {e}") from e
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise SystemExit("sentence-transformers が未インストールです: pip install -r requirements-rerank.txt")
    r = cfg["rerank"]
    return SentenceTransformerRerank(model=r["model"], top_n=r["top_n"], device=r.get("device"))


# ---------------------------------------------------------------- プロンプト

def load_prompt(cfg: dict) -> str:
    return (ROOT / "prompts" / f"{cfg['prompt']['template']}.txt").read_text(encoding="utf-8")


def build_context(nodes) -> str:
    from llama_index.core.schema import MetadataMode

    # LlamaIndex 既定のQAと同じく、メタデータ（file_name 等）込みの本文を並べる
    return "\n\n".join(n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes)


# ---------------------------------------------------------------- Ollama サーバログ監視
# Ollama は num_ctx を超えた入力を「エラーにせず」切り詰め、サーバログに
# `truncating input prompt` と出す。問ごとにログの増分を見て、切り詰めを確実に検出する。

def ollama_log_candidates() -> list[Path]:
    import os
    paths = []
    if os.environ.get("OLLAMA_LOG"):
        paths.append(Path(os.environ["OLLAMA_LOG"]))
    home = Path.home()
    paths += [
        home / ".ollama/logs/server.log",              # Ollama.app (macOS)
        Path("/opt/homebrew/var/log/ollama.log"),      # brew services (Apple Silicon)
        Path("/usr/local/var/log/ollama.log"),         # brew services (Intel)
    ]
    return [p for p in paths if p.exists()]


class OllamaLogWatcher:
    def __init__(self):
        cands = ollama_log_candidates()
        self.path = max(cands, key=lambda p: p.stat().st_mtime) if cands else None
        self.pos = self.path.stat().st_size if self.path else 0

    def new_lines(self) -> list[str]:
        if not self.path:
            return []
        size = self.path.stat().st_size
        if size < self.pos:  # ローテートされた
            self.pos = 0
        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(self.pos)
            data = f.read()
        self.pos = size
        return data.splitlines()

    def truncations(self) -> list[str]:
        return [l for l in self.new_lines() if "truncating input prompt" in l]


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()
