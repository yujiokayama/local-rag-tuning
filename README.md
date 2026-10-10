# local-rag-tuning

完全ローカル（Ollama + LlamaIndex）の日本語RAGを、**条件を1つずつ変えて比べる**ための実験キット。
**`git clone` して `./setup.sh` を叩けば、同じ実験を再現できる**ようにしてある。

- 前回の記事：[Mac mini（M4 Pro / 48GB）で『完全ローカル』のLLM + RAG環境を構築してみた](https://qiita.com/y-okayama-tb/items/0127a7c5203f44507474)
- このキットを使った記事（続編）：（公開後にリンクを追記）

> `data/docs/` の5文書と `eval/questions.yaml` の13問は、検証用に作った **架空の社内文書** です。実在の会社・製品・規程とは関係ありません。

- 生成: Ollama（qwen3:14b） / 埋め込み: Ollama（bge-m3） / フレームワーク: LlamaIndex / ベクトルDB: ChromaDB（ローカル永続化）
- 変えられるつまみ: コンテキスト長・チャンク分割・top_k・リランカー・ハイブリッド検索（BM25）・プロンプト・思考モード
- 1回の実行で「1問ごとの取得チャンク・回答・トークン数・所要時間・実行環境」が全部 `results/` に残る

---

## 0. 前提

| 項目 | 想定 |
|---|---|
| マシン | Apple Silicon Mac（M4 Pro / 48GB で設計。32GB 以上推奨） |
| OS | macOS（Linux でも Ollama を自分で入れてあれば動く） |
| 必要なもの | Homebrew、空きディスク約 20GB（モデル + torch） |
| ネットワーク | **セットアップ時だけ**必要（モデルとライブラリのDL）。実験中は完全オフラインで動く |

## 1. 入手

```bash
git clone https://github.com/<ユーザー名>/local-rag-tuning.git
cd local-rag-tuning
```

架空の社内文書5本（`data/docs/`）と13問（`eval/questions.yaml`）は同梱済み。
自分の文書で試すときは、次の場所を差し替える：

```
data/docs/            ← 文書（.md / .txt / .pdf など）をそのまま置く
eval/questions.yaml   ← 質問を書く（書式は eval/questions.sample.yaml）
```

> ⚠️ 会社の文書を個人PCや公開リポジトリに置かないこと。

## 2. セットアップ（1コマンド）

```bash
cd local-rag-tuning
./setup.sh
```

やっていること（何度実行しても安全。入っているものはスキップ）：

1. Ollama が無ければ `brew install ollama`、サーバが止まっていれば `brew services start ollama`
   （Ollama.app で入れている場合はアプリを起動しておけばそのまま使う）
2. `qwen3:14b` と `bge-m3` を pull（済みならスキップ）
3. Python 3.10+ で `.venv` を作り、`requirements.txt`（バージョン固定）を導入
4. リランカー用に `sentence-transformers`（torch込み）を導入し、`BAAI/bge-reranker-v2-m3` を事前DL
5. `src/check_env.py` で動作確認

オプション：

```bash
WITH_RERANK=0 ./setup.sh        # torch を入れない（リランク実験 30/42 以外は動く）
LLM=gemma3:12b ./setup.sh       # 別モデルも pull（base.yaml の llm も書き換える）
```

最後にこう出ればOK：

```
✅ Ollama 0.xx @ http://localhost:11434
✅ model qwen3:14b  {...}
✅ model bge-m3  {...}
✅ embedding dim=1024
✅ LLM 応答: 'OK' (...)
   LlamaIndex が送る num_ctx = 40960 ...
セットアップOK。
```

## 3. 動作確認の順番

```bash
make mock                         # ① Ollama を使わず配線だけ確認（数秒）
make run EXP=00_baseline SAMPLE=1 # ② 同梱サンプル文書3本・4問で実モデルを通す
make run EXP=00_baseline          # ③ 本番の文書・13問でベースライン
```

## 4. 実験の回し方

| コマンド | 内容 |
|---|---|
| `make run EXP=30_rerank` | 1つの実験を実行 |
| `make all` | `config/experiments/*.yaml` を全部順に実行 |
| `make chunks EXP=22_chunk512_sentence` | チャンク分割結果を `results/chunks/` に書き出して目視 |
| `make compare` | 実験ごとの最新結果を Markdown 表で出力（Qiita に貼れる） |
| `make check` | 環境チェック |
| `make clean-index` | 埋め込みキャッシュ（`.chroma/`）を消す |

一部の質問だけ回したいとき：`.venv/bin/python src/rag_eval.py 30_rerank --only q01,q05`

### 実験一覧（`config/experiments/`）

| ファイル | 変えるもの | 見たいこと |
|---|---|---|
| 00_baseline | なし（base.yaml = 前回記事の構成: 日本語分割 512/64, top_k=4） | 基準 |
| 01_ctx8192 / 02_ctx16384 / 03_ctx3900 | コンテキスト長 | 切り詰めの有無・速度・メモリ |
| 10_topk1 / 11_topk8 | top_k | 前回の「質問種別で最適 top_k が割れる」の再確認 |
| 20_chunk1024_default | LlamaIndex 既定の分割（1024/200） | 既定のままだと何が起きるか（5文書=5チャンク） |
| 21_chunk256 | チャンクサイズ半分 | 単一/複数文書型への影響 |
| 22_chunk512_sentence | 分割器だけ既定（sentence）に戻す | 日本語句読点優先の分割の効果 |
| 30_rerank | 候補20件 → リランク4件 | top_k のジレンマが解けるか（本命） |
| 40_hybrid / 41_hybrid_bigram | BM25併用（形態素 / 文字2-gram） | 固有名詞の取りこぼし対策 |
| 42_hybrid_rerank | ハイブリッド候補 → リランク | 組み合わせ |
| 50_prompt_strict | 「資料にないなら答えない」プロンプト | もっともらしい誤答が減るか |
| 60_no_thinking | qwen3 の思考をオフ | 速度と回答の変化 |
| 70_best_combo | チャンク256 + リランク + 思考オフ | 良かったものの組み合わせ |

実験ファイルは **base.yaml との差分だけ** を書く。新しい実験は yaml を1枚足すだけ。
01〜03 でコンテキスト長を決めたら、10番以降の `context_window: 8192` を決めた値に揃えること。

## 5. 出力

```
results/
├── summary.csv                       … 全実行の集計（1実行 × all/single/multi の3行）
├── chunks/<実験名>.md                … make chunks の出力
└── <実験名>/<日時>/
    ├── results.jsonl                 … 1問ごと: 候補/最終チャンク, 回答, prompt_tokens, 時間, 切り詰めログ
    ├── summary.json
    ├── config.yaml                   … 実行時の設定（再現用）
    └── env.json                      … マシン, Ollama版, モデルdigest/量子化, ライブラリ版
```

指標：

| 指標 | 意味 |
|---|---|
| MRR | 期待ファイルが最初に出てきた順位の逆数の平均（**検索比較の主指標**） |
| hit@1 | 1位のチャンクが期待ファイルだった割合 |
| 根拠hit | 期待キーワードが「期待ファイル由来の」上位 k 件チャンクに全部入った割合（答えの箇所まで取れたか。キーワード無しの問は除外） |
| hit_all | 期待ファイルが **すべて** 上位 k 件に入った割合（`eval.hit_k`）。本番5文書=8チャンクでは k=4 でほぼ満点になるので参考値 |
| recall | 期待ファイルのうち上位 k 件に入った割合の平均 |
| 正答率 | `expected_keywords` を回答がすべて含んだ割合（目安。最終判断は results.jsonl の回答を目視。q12 は正解キーワードが質問文にも含まれるため、質問のオウム返しでも正解扱いになる点に注意） |
| ctx満杯疑い | 入力が num_ctx で切り詰められた（疑い）問の数 |
| 生成打ち切り | 生成が `num_predict` の上限に達した／タイムアウト等で失敗した問の数 |

> 前回記事の `file_hit@4` と定義を合わせたい場合は `rag_eval.py` の `score_question()` を調整する。

計測マシン名は自動で「チップ / メモリ」になる。記事用に名前を付けたいときは `RAG_MACHINE="Mac mini M4 Pro 48GB" make run ...`。

## 6. 仕込んである「静かに劣化する設定」対策

前回記事で「エラーが出ないまま精度だけ落ちる」と書いた部分を、計測で見えるようにしてある。

1. **コンテキスト長の切り詰め**
   Ollama は `num_ctx` を超えた入力を **エラーにせず切り詰める**（サーバログに `truncating input prompt` が出る）。
   `rag_eval.py` は問ごとに Ollama のサーバログの増分を見て、切り詰めが起きた問に `⚠ctx満杯の疑い` を付ける。
   ログの場所は `~/.ollama/logs/server.log`（Ollama.app）か `/opt/homebrew/var/log/ollama.log`（brew services）を自動で探す。
   見つからない場合は `OLLAMA_LOG=/path/to/log` で指定（指定がなければトークン数からの推定だけになる）。

   また、LlamaIndex が Ollama に送る `num_ctx` は **ライブラリのバージョンで既定値が違う**。
   固定している llama-index-llms-ollama 0.11.0 では、`context_window` 未指定だとモデルの最大長（qwen3 なら 40960）を送る。
   古い版では 3900 が既定だった。実際の値は `make check` と各実行の先頭行 `[llm] ... num_ctx=` で確認できる。

2. **日本語のチャンク分割**
   `make chunks` で分割結果を全文ダンプし、「文末が句点で終わっていないチャンク数」を出す。

3. **埋め込みキャッシュの誤用**
   ChromaDB のコレクション名を「埋め込みモデル + 分割設定 + 文書内容」のハッシュにしているので、
   チャンク設定を変えると自動で別インデックスになる（古い埋め込みを読み続ける事故が起きない）。

4. **PC固有の絶対パスの混入**
   LlamaIndex の既定では `file_path`（`/Users/<名前>/...`）が埋め込み・プロンプトの両方に入る。
   埋め込みは本文のみ、LLM には `file_name` だけを渡すようにしている（`common.py` の `EMBED_METADATA_KEYS` / `LLM_METADATA_KEYS`）。

5. **思考の暴走**
   qwen3 の思考モードを temperature=0 で使うと、思考が終わらず `request_timeout` まで回り続けることがある
   （モデルカードでも思考モードでの greedy decoding は非推奨）。
   `ollama.num_predict`（既定 4096）で生成を打ち切り、その問を「生成打ち切り」として記録して次へ進む。

6. **再現性**
   temperature=0・seed 固定。ライブラリはバージョン固定。実行ごとに設定と環境を保存。

## 計測結果（参考）

`results/` に、Mac mini M4 Pro 48GB で計測した全実験の結果（1問ごとの回答を含む）を同梱している。
比較表は `results/compare.md`。考察は続編の記事を参照。

## 7. 構成

```
local-rag-tuning/
├── setup.sh                 … セットアップ一式
├── Makefile
├── requirements.txt         … バージョン固定
├── requirements-rerank.txt  … リランカー用（torch）
├── config/
│   ├── base.yaml            … 全設定の既定値（コメントに説明あり）
│   └── experiments/*.yaml   … 実験ごとの差分
├── prompts/                 … default（LlamaIndex既定相当）/ strict
├── data/docs/               … ← 本番の文書を置く
├── data/docs.sample/        … 動作確認用の架空文書3本
├── eval/questions.yaml      … ← 本番の質問を書く
├── eval/questions.sample.yaml
└── src/
    ├── rag_eval.py          … 評価本体
    ├── common.py            … 分割・インデックス・BM25・RRF・リランク・ログ監視
    ├── check_env.py         … 環境チェック
    ├── dump_chunks.py       … チャンク目視用
    ├── compare.py           … 結果の Markdown 表
    └── envinfo.py           … 実行環境の記録
```

## 8. トラブルシューティング

| 症状 | 対処 |
|---|---|
| `Ollama に接続できません` | `brew services start ollama` か Ollama.app を起動。`curl localhost:11434/api/version` で確認 |
| `model ... がありません` | `ollama pull qwen3:14b` / `ollama pull bge-m3` |
| 文書が見つからない / 質問が0件 | `data/docs/` と `eval/questions.yaml` を用意する。動作確認だけなら `SAMPLE=1` |
| 回答に思考過程が混ざる・遅い | `base.yaml` の `ollama.thinking: false`（qwen3 の思考をオフ）。`<think>` タグは集計前に除去している |
| リランク実験で `sentence-transformers が未インストール` | `.venv/bin/pip install -r requirements-rerank.txt` |
| メモリが足りない | `02_ctx16384` は KV キャッシュが大きい。`ollama ps` で常駐モデルを確認し、不要なモデルを `ollama stop <model>` |
| 設定を変えたのに結果が同じ | `[index]` 行のコレクション名が変わっているか確認。怪しければ `make clean-index` |

## ライセンス

MIT License（[LICENSE](LICENSE)）
