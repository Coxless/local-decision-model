# フェーズ 6 の結果: ローカルサーバー (開発 PC 分)

2026-10-11。System One (TypeSafe の jev) の HTTP API と同じ形で答えるサーバーを足した。
NPU PC が要る作業と、小型の LLM との比較は残っている。

## 1. 作ったもの

| もの | 場所 |
|---|---|
| サーバー (`POST /v1/systemone`、`GET /v1/models`、`GET /health`) | `src/local_decision_model/server.py` |
| `serve` コマンド | `cli.py` (`--host 127.0.0.1 --port 8642` が既定) |
| `criteria`、`confidence`、状態の JSON 文字列化、回答の形 | `schema.py` |
| `serve` の action | `.workshop/gpu.yaml`、`.workshop/npu.yaml`、`scripts/npu.ps1` |
| テスト | `tests/test_server.py`、`tests/test_schema.py` |

依存に `fastapi` と `uvicorn` を足した。テスト用に `httpx` と公式 SDK の `typesafe-sdk` を dev に足した。

## 2. API の形の確認

公式 SDK (`typesafe-sdk` 0.7.4) は、TypeSafe の OpenAPI から生成したスキーマ (`typesafe_sdk/_schemas/models.py`) を
持っている。プランで「要確認」だった点は、これで確かめた。

- パスは `POST /v1/systemone` と `GET /v1/models` (`{"models": [{name, description, release_date}]}`)
- 検証エラー (`422`) は `{"detail": [{"loc", "msg", "type", "input"}]}`。`loc` は
  `["body", "questions", <質問名>, <型>, "criteria"]` のように、質問名の後ろに型が入る。FastAPI の既定と同じ形
- `instructions` は省ける (noul は `criteria` だけでもよい)

リクエストは同じ形の pydantic モデルで検証するので、形の間違いは同じ `422` になる。
未対応のもの (score、オブジェクトや配列で書いた `instructions` / `criteria`) と未知の `model` も、同じ形の `422` で返す。
未知の `model` に TypeSafe が何を返すかは確かめていない。

## 3. 質問から仮説文への変換

| リクエスト | 仮説文 |
|---|---|
| noul、`instructions` だけ | `instructions` そのまま |
| noul、`criteria.true` あり | `criteria.true` (`instructions` と `criteria.false` は使わない) |
| choice、`criteria` の値が null | `instructions` + 空白 + 選択肢 (`{option}` があれば置き換え) |
| choice、`criteria` の値が説明 | 選択肢の代わりに「選択肢 (説明)」を入れる |

**精度は測っていない**。学習データの choice はどれも `{option}` 入りの平叙文で、説明もないので、
jev の書き方 (疑問文 + `criteria` の説明) は学習にない形になる。フェーズ 0 の zero-shot では「疑問文 + 選択肢」の
連結で正解率が落ちた (0.75 → 0.40)。学習後のモデルでどうなるかは、プランの「精度の確認」で測る。

## 4. 動作確認

`models/packed` (torch / cuda) と `models/packed-ov` (OpenVINO / CPU) でサーバーを起動し、同じリクエスト
(noul 4 問 + choice 2 問) を送った。回答は両方で同じ (小数 3 桁まで) だった。

| | torch / cuda | OpenVINO / CPU |
|---|---|---|
| 起動してから `/health` が 200 になるまで | 測っていない | 約 6 秒 (系列長 3 種類のコンパイルを含む) |
| noul 1 問の往復 (curl、HTTP を含む) | 約 25 ms | 約 50 ms |

- 公式 SDK の `TypeSafeClient(base_url=...)` で、`system_one` と `models.list()` が通る
- 読み込み中は `/health` と `/v1/systemone` が `503` を返す (SDK は 5xx を自動で再試行する)
- ワークショップの中の `127.0.0.1` は、ホストから見えない。action は `--host 0.0.0.0` を付け、
  ホストからは `http://gpu.<プロジェクトのディレクトリ名>.wp:8642` でつなぐ。Windows の NPU PC はコンテナがないので
  既定 (`127.0.0.1`) のまま

速さは数回の curl で見ただけで、`bench` のような計測はしていない。

## 5. プランから変えた点

- `usage.input_tokens` は「状態と仮説のトークン数の合計」にした。切り詰めや、パックごとの状態の繰り返しは数えない
- `decide` コマンドの出力も System One と同じ形にした (`value` / `probability` → `noul`、`choice` / `confidence`)
- ワークショップの action は `0.0.0.0` で待ち受ける (上の理由)。プランの「既定は localhost のみ」は CLI の既定として守っている

## 残っていること

- (比較) TypeSafe 互換のローカルサーバーで小型の LLM を動かし、同じリクエストで精度と速さを比べる
- jev の書き方 (疑問文の choice、`criteria` の説明、JSON の状態) での精度を測る
- score、オブジェクトや配列で書いた `instructions` / `criteria`
- NPU PC: `scripts\npu.ps1 serve` で起動して確かめる
