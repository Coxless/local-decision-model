# local-decision-model: jev ライクな判断モデルを NPU で動かす

[TypeSafe の jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)（System One model）と同じ考え方で、
**「自由テキストの状態 + 名前付きの質問 → 型付き・較正された確率付きの回答」** を返す小さなモデルを
ローカルで動かすための開発リポジトリです。jev 本体は重みが公開されていない API 専用モデルなので、
ここでは小型のエンコーダ（多言語 NLI クロスエンコーダ）で同じインターフェースを再現します。

- 開発 PC（NVIDIA GPU）: 学習・評価・OpenVINO IR へのエクスポート → ワークショップ `gpu`
- NPU 搭載 Intel PC（Core Ultra）: OpenVINO で NPU 推論 → ワークショップ `npu`

開発環境は [Canonical Workshop](https://ubuntu.com/workshop/docs/) で定義しています（`~/development/template` がベース）。

## 仕組み

```
状態: "この本はとても楽しかった…"      質問: positive = "The book review is positive."
        │                                       │
        └──────── (状態, 仮説) のペア ─────────┘
                        │
          多言語 NLI クロスエンコーダ (mDeBERTa-v3-base, 約 2.8 億パラメータ)
                        │
     判断ロジット z = logit[entailment] - logit[contradiction]
                        │
     noul         : P = sigmoid(z / T)
     choice       : 選択肢ごとの z を softmax(z / T)
```

- 文字列は生成しないので、出力が型から外れることはありません（jev と同じ方針）。
- 温度 `T` は学習後に検証データで較正します（`train` が自動で行い `s1.json` に保存）。
- ベースモデル: [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)（MIT, 日本語対応）。学習しなくても zero-shot で動きます。
- NPU は動的形状が苦手なので、入力を `[batch_size, max_length]`（既定 8 × 256）に固定してコンパイルします。

## 構成

```
local-decision-model/
├── .workshop/
│   ├── gpu.yaml              # 開発 PC 用ワークショップ (CUDA)
│   ├── npu.yaml              # NPU PC 用ワークショップ (OpenVINO)
│   ├── nvidia-gpu/           # プロジェクト内 SDK: NVIDIA GPU を渡す
│   └── intel-npu/            # プロジェクト内 SDK: NPU / Intel GPU を渡し、ドライバを入れる
├── src/local_decision_model/
│   ├── schema.py             # 質問 (Noul / Choice) と回答の型
│   ├── decide.py             # 状態 + 質問 → 回答
│   ├── scoring.py            # トークナイズ、判断ロジット、s1.json
│   ├── calibration.py        # 温度較正・ECE
│   ├── packing.py            # パック方式: 状態と質問を 1 本の系列に詰める
│   ├── packed_model.py       # パック方式のモデル (1 回の推論で複数の質問に答える)
│   ├── torch_backend.py      # PyTorch (CUDA / CPU)
│   ├── openvino_backend.py   # OpenVINO (NPU / GPU / CPU)
│   ├── export.py             # PyTorch → OpenVINO IR
│   ├── distill.py            # 学習データの形式と、ペア方式のモデルによるラベル付け (比較用)
│   ├── train.py              # ファインチューニング
│   └── cli.py                # python -m local_decision_model ...
├── examples/review.yaml      # リクエスト例
├── data/
│   ├── states.jsonl          # 学習用の状態 (scripts/phase2/generate_states.py で Claude Code に書かせた問い合わせ文)
│   ├── questions.yaml        # 学習用の質問セット
│   ├── train-pair.jsonl      # zero-shot のペア方式のラベル (比較用。学習には使わない): distill の出力
│   ├── train-claude.jsonl    # 学習データ。教師 (Claude Code) のラベル: scripts/phase2/label_with_claude.py の出力
│   ├── eval-states.jsonl     # 評価セット用の状態
│   ├── eval-draft.jsonl      # 評価セットの下書き (Claude Code のラベル。人の見直しはまだ)
│   ├── sample-states.jsonl   # distill の入力例: 状態
│   ├── sample-questions.yaml # distill の入力例: 質問セット
│   ├── sample.jsonl          # distill の出力例 (1 行 = 1 つの状態 + 複数の質問 + labels)
│   └── sample-pairs.jsonl    # train の学習データの例 (1 行 1 ペア)
└── tests/
```

## 前提条件 (両方の PC)

テンプレートと同じです。

```bash
sudo snap install --channel=6/stable lxd
sudo snap install --classic workshop
sudo lxd init --minimal
sudo usermod -aG lxd "$USER"   # 反映には再ログインが必要
```

NPU PC ではさらに、**ホスト側**で NPU のカーネルドライバ（`intel_vpu`）とファームウェアが動いている必要があります。
Ubuntu 24.04 以降の標準カーネルなら通常は入っています。

```bash
ls /dev/accel/accel0          # これがあれば OK
lsmod | grep intel_vpu
```

ユーザー空間ドライバ（Level Zero / NPU コンパイラ / Intel GPU ランタイム）はワークショップ内に自動で入ります
（`.workshop/intel-npu/hooks/setup-base`）。

## 開発 PC (NVIDIA GPU) での使い方

```bash
workshop launch gpu
workshop run gpu -- setup      # uv sync --extra cuda --extra openvino
workshop run gpu -- check      # GPU が見えるか確認
workshop run gpu -- decide     # examples/review.yaml に回答 (初回はモデルをダウンロード)
workshop run gpu -- bench      # レイテンシ計測
```

学習（ファインチューニング + 温度較正）と NPU 用のエクスポート:

```bash
workshop run gpu -- train data/train.jsonl --out models/finetuned
workshop run gpu -- decide --model models/finetuned
workshop run gpu -- export --model models/finetuned --out models/finetuned-ov
```

`train` の学習データは 1 行 1 ペアの JSONL です（`data/sample-pairs.jsonl` 参照）。`label` は 0/1 のほか 0〜1 の確率も使えるので、
大きな LLM が付けた確率を教師にした蒸留もできます。Choice は「instructions + 選択肢」の仮説文に展開して、
正解の選択肢を 1、それ以外を 0 にした行で学習します。

```json
{"state": "This book was a delight to read.", "question": "The book review is positive.", "label": 1}
```

蒸留用の学習データは、教師の Claude Code がラベルを付けます（`scripts/phase2/label_with_claude.py`、ワークショップの外で実行）。
`distill` は、同じ形式のラベルをペア方式のモデルで付けます（教師のラベルと比べるため）:

```bash
workshop run gpu -- distill data/sample-states.jsonl --questions data/sample-questions.yaml --out data/train.jsonl
```

出力は 1 行が「1 つの状態 + 複数の質問 + `labels`」の JSONL です（`data/sample.jsonl` 参照）。`labels` は noul が確率、
choice が選択肢ごとの確率です。パック方式の学習（フェーズ 3）はこの形式を読みます。`train` はまだ 1 行 1 ペアの形式だけを読みます。

```json
{"state": "…", "questions": {"refund": "顧客は返金を求めていますか？", "topic": {"type": "choice", "instructions": "…{option}…", "options": ["配送", "品質"]}},
 "labels": {"refund": 0.94, "topic": {"配送": 0.81, "品質": 0.19}}}
```

ベースモデルのままエクスポートする場合は `workshop run gpu -- export`（出力先 `models/base-ov`）。

## NPU PC (Intel Core Ultra) での使い方

```bash
git clone <このリポジトリ> && cd local-decision-model
workshop launch npu
workshop connect npu/intel-npu:npu :custom-device   # 初回のみ (NPU は自動接続されない)
workshop run npu -- setup      # uv sync --extra openvino
workshop run npu -- check      # /dev/accel と OpenVINO のデバイス一覧に NPU があるか確認
```

モデル（`models/` は Git に含めない）は開発 PC から転送します。

```bash
# 開発 PC で
rsync -av models/base-ov/ <npu-pc>:~/development/local-decision-model/models/base-ov/
```

```bash
workshop run npu -- decide                                  # NPU で推論
workshop run npu -- decide --model models/finetuned-ov
workshop run npu -- bench                                   # NPU のレイテンシ
workshop run npu -- bench --device GPU                      # 比較: 内蔵 GPU
workshop run npu -- bench --device CPU                      # 比較: CPU
```

NPU の初回コンパイルは時間がかかるので、結果を `.cache/openvino` にキャッシュしています。

## 出力例

```json
{
  "answers": {
    "positive":   {"type": "noul", "value": true, "probability": 0.97},
    "recommends": {"type": "noul", "value": true, "probability": 0.81},
    "genre":      {"type": "choice", "value": "a novel",
                   "probabilities": {"a novel": 0.93, "a cookbook": 0.02, "a science textbook": 0.05}}
  },
  "latency_ms": 12.3
}
```

## actions

| action   | gpu | npu | 内容 |
| -------- | --- | --- | ---- |
| `setup`  | ✓ | ✓ | 依存パッケージの取得 |
| `check`  | ✓ | ✓ | デバイス確認 |
| `test`   | ✓ | ✓ | pytest（引数はそのまま渡す） |
| `lint` / `fmt` | ✓ |  | ruff |
| `decide` | ✓ | ✓ | リクエスト YAML に回答（既定 `examples/review.yaml`） |
| `bench`  | ✓ | ✓ | レイテンシ計測 |
| `distill` | ✓ |  | 状態と質問セットにペア方式の確率を付ける（比較用） |
| `train`  | ✓ |  | ファインチューニング + 温度較正 |
| `export` | ✓ |  | OpenVINO IR に変換 |

`decide` / `bench` の引数は後ろに足せます（例: `workshop run npu -- bench --batch-size 4 --runs 500`）。

## ワークショップについての注意

- ワークショップが 2 つあるので、`workshop shell gpu` のように名前を指定します。
- venv はワークショップごとに `~/.venv`（`UV_PROJECT_ENVIRONMENT`）に作ります。`/project/.venv` は使いません。
- Hugging Face のキャッシュは `/project/.cache/huggingface`（`HF_HOME`）に置き、refresh しても消えないようにしています。
- `gpu` プラグは system SDK には付けられないため、プロジェクト内 SDK（`.workshop/nvidia-gpu/`, `.workshop/intel-npu/`）で宣言しています。
  定義からは `project-<名前>` で参照します。
- NPU の custom-device プラグは `connections:` に書いても自動接続されないため、`workshop connect` で手動接続します。
- NPU ドライバのバージョンは `.workshop/intel-npu/hooks/setup-base` の `NPU_DRIVER_VERSION` / `NPU_DRIVER_BUILD` で変えて、`workshop refresh npu` します。
- `.workshop.lock` はコミットしません（`.gitignore` 登録済み）。

## 今後

- パック方式の学習: 教師 (Claude Code) のラベル `data/train-claude.jsonl` で蒸留する（`docs/plan.md` のフェーズ 3）。
- INT8 量子化（NNCF）で NPU のレイテンシをさらに下げる。
- 数値・リストなど noul / choice 以外の型。
