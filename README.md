# local-s1: jev ライクな判断モデルを NPU で動かす

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
          多言語 NLI クロスエンコーダ (MiniLM L6, 約 1 億パラメータ)
                        │
     判断ロジット z = logit[entailment] - logit[contradiction]
                        │
     bool  (Noul) : P = sigmoid(z / T)
     choice       : 選択肢ごとの z を softmax(z / T)
```

- 文字列は生成しないので、出力が型から外れることはありません（jev と同じ方針）。
- 温度 `T` は学習後に検証データで較正します（`train` が自動で行い `s1.json` に保存）。
- ベースモデル: [`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli)（MIT, 日本語対応）。学習しなくても zero-shot で動きます。
- NPU は動的形状が苦手なので、入力を `[batch_size, max_length]`（既定 8 × 256）に固定してコンパイルします。

## 構成

```
local-s1/
├── .workshop/
│   ├── gpu.yaml              # 開発 PC 用ワークショップ (CUDA)
│   ├── npu.yaml              # NPU PC 用ワークショップ (OpenVINO)
│   ├── nvidia-gpu/           # プロジェクト内 SDK: NVIDIA GPU を渡す
│   └── intel-npu/            # プロジェクト内 SDK: NPU / Intel GPU を渡し、ドライバを入れる
├── src/local_s1/
│   ├── schema.py             # 質問 (Noul / Choice) と回答の型
│   ├── decide.py             # 状態 + 質問 → 回答
│   ├── scoring.py            # トークナイズ、判断ロジット、s1.json
│   ├── calibration.py        # 温度較正・ECE
│   ├── torch_backend.py      # PyTorch (CUDA / CPU)
│   ├── openvino_backend.py   # OpenVINO (NPU / GPU / CPU)
│   ├── export.py             # PyTorch → OpenVINO IR
│   ├── train.py              # ファインチューニング
│   └── cli.py                # python -m local_s1 ...
├── examples/review.yaml      # リクエスト例
├── data/sample.jsonl         # 学習データの形式例
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

学習データは 1 行 1 ペアの JSONL です（`data/sample.jsonl` 参照）。`label` は 0/1 のほか 0〜1 の確率も使えるので、
大きな LLM が付けた確率を教師にした蒸留もできます。Choice は「instructions + 選択肢」の仮説文に展開して、
正解の選択肢を 1、それ以外を 0 にした行で学習します。

```json
{"state": "This book was a delight to read.", "question": "The book review is positive.", "label": 1}
```

ベースモデルのままエクスポートする場合は `workshop run gpu -- export`（出力先 `models/base-ov`）。

## NPU PC (Intel Core Ultra) での使い方

```bash
git clone <このリポジトリ> && cd local-s1
workshop launch npu
workshop connect npu/intel-npu:npu :custom-device   # 初回のみ (NPU は自動接続されない)
workshop run npu -- setup      # uv sync --extra openvino
workshop run npu -- check      # /dev/accel と OpenVINO のデバイス一覧に NPU があるか確認
```

モデル（`models/` は Git に含めない）は開発 PC から転送します。

```bash
# 開発 PC で
rsync -av models/base-ov/ <npu-pc>:~/development/local-s1/models/base-ov/
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
    "positive":   {"type": "bool", "value": true, "probability": 0.97},
    "recommends": {"type": "bool", "value": true, "probability": 0.81},
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

- 教師データ作り: 大きな LLM（[system-one-adapter](https://github.com/typesafe-ai/system-one-adapter-python) など）で
  状態と質問に確率を付けさせ、`train` で蒸留する。
- INT8 量子化（NNCF）で NPU のレイテンシをさらに下げる。
- 数値・リストなど bool / choice 以外の型。
