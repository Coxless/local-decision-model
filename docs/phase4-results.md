# フェーズ 4 の結果 (エクスポートと OpenVINO 推論)

実施: 2026-10-11。環境: 開発 PC (ワークショップ `gpu`、GTX 1660 Ti、Core i7-10700F)。
1〜3 節は開発 PC でできる範囲 (torch の GPU と OpenVINO の CPU) だけを行ったときの記録。
2026-10-24 に NPU PC で確認した分を 4 節に追記した。

## まとめ

- パック方式のモデルを OpenVINO IR にできるようにした (`export`。方式は `s1.json` で決まる)
- OpenVINO でパック方式を推論できるようにした (`OpenVINOPackedScorer`)。系列長 128 / 256 / 512 ごとに
  静的な形状でコンパイルし、パックの長さに合うものを使う
- **torch と OpenVINO の CPU で結果が一致した**。評価セットの正解率は noul 0.934、choice 0.839 (torch は 0.934 / 0.838)
- OpenVINO の CPU での速さは、パック方式が 77〜392 ms、ペア方式が 1.7〜5.1 秒 (2 節)
- IR を静的な形状にすると、形状が動的な演算も、条件で枝が変わる `If` も残らない (3 節)
- NPU では、系列長 128 と 512 のコンパイルと推論ができ、答えは OpenVINO の CPU と一致した。
  ただし系列長 160〜480 で NPU がハングし、内蔵 GPU のほうが 6〜7 倍速い (4 節)

## 1. 作ったもの

| ファイル | 内容 |
|---|---|
| `export.py` | パック方式の変換 (`convert_packed`) を追加。IR の入力は `input_ids` / `question_ids` / `option_ids` / `positions` (`[B, L]`) と `marker_index` (`[B, M]`)、出力は `decision_logits` (`[B, M]`) |
| `openvino_backend.py` | `OpenVINOPackedScorer` を追加 |
| `cli.py` | `decide` / `bench` / `evaluate` の `--backend openvino` がパック方式の IR を読む。`bench` は複数のリクエストを順に測る |
| `tests/test_openvino_packed.py` | 小さな DeBERTa で、torch と OpenVINO の CPU の判断ロジットが一致すること (系列長 3 種類、パックの分割、詰め方の違い) |
| `examples/bench/*.yaml` | `bench` の条件 (noul 1 / 3 / 10 問、選択肢 20 個の choice) |
| `scripts/phase4/ir_ops.py` | IR の演算の種類を一覧にする (3 節) |

- IR は役割埋め込み、見える範囲と相対位置の計算、判断ロジットへの変換までを含む。
  モデルディレクトリに `role_embeddings.safetensors` は要らない
- IR は 1 つだけ保存する (形状は動的)。系列長ごとの形状は、読み込み側で `reshape` して決める。
  `models/packed-ov` は 566 MB (重みは FP16)
- 系列長ごとのコンパイルは、その長さを初めて使うときに行う。まとめて済ませるときは `compile_all` を呼ぶ
  (サーバーの起動時に使う想定)
- 出力の名前はプランの `logits` から `decision_logits` に変えた (分類ロジットではなく判断ロジットなので)

## 2. torch との一致と速さ

評価セット (`data/eval-draft.jsonl`、300 件 × 41 問)。温度は `s1.json` のもの。

| バックエンド | 型 | 正解率 | NLL | ECE |
|---|---|---|---|---|
| torch (GPU、FP32) | noul | 0.9335 | 0.2931 | 0.0172 |
| OpenVINO (CPU、FP16 の重み) | noul | 0.9335 | 0.2931 | 0.0172 |
| torch (GPU、FP32) | choice | 0.8380 | 0.9473 | 0.0260 |
| OpenVINO (CPU、FP16 の重み) | choice | 0.8387 | 0.9474 | 0.0267 |

choice の差は 1,500 件のうち 1 件 (`urgency`) の判定が変わった分。`examples/review.yaml` の確率の差は 1e-4 未満。

速さ (p50、ミリ秒)。`workshop run gpu -- bench examples/bench/*.yaml --model <モデル>` で測った。
状態は 105 トークンほどの問い合わせ文で、どの条件も 1 つのパックに入る。

| 条件 | パックの系列長 | パック方式 torch GPU | ペア方式 torch GPU | パック方式 OpenVINO CPU | ペア方式 OpenVINO CPU |
|---|---|---|---|---|---|
| noul 1 問 | 128 | 27 | 39 | 77 | 1,687 |
| noul 3 問 | 256 | 37 | 94 | 153 | 1,692 |
| noul 10 問 | 256 | 38 | 260 | 171 | 3,410 |
| choice (選択肢 20 個) | 512 | 75 | 499 | 392 | 5,063 |

- 回数は torch が 100 回、OpenVINO のパック方式が 20 回、ペア方式が 10 回。トークナイズの時間を含む
- ペア方式の OpenVINO は、形状を 8 × 288 に固定している (ペアが 1 つでも 8 つ分を計算する)。
  このためペアが 8 つ以下なら 1.7 秒、選択肢 20 個では 3 回分で 5 秒かかる
- パック方式は質問が増えても 1 回の推論で済み、系列長が同じなら時間はほとんど変わらない
- CPU でのコンパイルは系列長 1 つにつき数秒 (`decide` の初回は 4.2 秒)。NPU ではもっとかかるはず

## 3. IR の演算

`scripts/phase4/ir_ops.py` で、NPU と同じ静的な形状 (パック方式は 1 × 512、ペア方式は 8 × 256) にして
定数を畳み込んだ後の演算を数えた。右はフェーズ 0 で作ったベースモデルのペア方式の IR。

| 演算 | パック方式 | ペア方式 |
|---|---|---|
| Abs | 1 | |
| Add | 174 | 161 |
| BitwiseAnd | 2 | 1 |
| BitwiseNot | 1 | 1 |
| BitwiseOr | 2 | |
| Broadcast | 26 | 24 |
| Ceiling | 1 | 1 |
| Clamp | 24 | 13 |
| Convert | 34 | 29 |
| Divide | 38 | 38 |
| Equal | 4 | |
| Gather | 5 | 2 |
| GatherElements | 25 | 24 |
| Gelu | 13 | 13 |
| Greater | 3 | |
| GreaterEqual | 1 | |
| If | 12 | 12 |
| Less | 1 | |
| LessEqual | 1 | 1 |
| Log | 1 | 1 |
| MVN | 25 | 25 |
| MatMul | 122 | 122 |
| Multiply | 40 | 41 |
| Reshape | 123 | 120 |
| Select | 14 | 14 |
| Sign | 1 | |
| Slice | | 1 |
| Softmax | 12 | 12 |
| Squeeze | 24 | 14 |
| Subtract | 2 | |
| Transpose | 72 | 72 |
| Unsqueeze | 10 | 6 |

- どちらも、形状が動的な演算は 0、条件が定数でない `If` は 0。`If` は DeBERTa のアテンションの中の分岐で、
  12 個とも条件が定数になる (コンパイル時にどちらかの枝に置き換えられる)
- パック方式だけにある演算は、見える範囲を作る比較 (`Equal`、`Greater`、`GreaterEqual`、`BitwiseOr`) と、
  相対位置を入力から計算する部分 (`Abs`、`Sign`、`Less`、`Subtract`)。ペア方式では相対位置が定数に畳み込まれる
- パック方式は相対位置を入力から計算するので、対数バケットの計算 (`Log`、`Ceiling`、`Clamp`) と
  `GatherElements` が実行時に動く。NPU で問題が出るとしたら、まずここを疑う

## 4. NPU PC での確認 (2026-10-24 追記)

2026-10-24 に NPU PC (Windows 11、Core Ultra 5 125H、NPU ドライバ 32.0.100.5540、OpenVINO 2026.4.1) で
`models/packed-ov` を動かした。使ったのは学習済みのパック方式の IR (HF の `Coxless/local-decision-model` の
`packed-ov/`)。

**結論**: パック方式の IR は NPU でコンパイルでき、答えは OpenVINO の CPU と一致した。
ただし**系列長 160〜480 では推論の実行時に NPU がハングする**ため、使える長さは 128 と 512 だけだった。
速さは内蔵 GPU が NPU の 6〜7 倍だった。

### 4.1 環境づくりで詰まった点

- `requires-python = ">=3.12"` だけだと uv が Python 3.15 を選び、openvino の wheel (cp312〜cp314) が
  無いので `setup` が失敗する。`.python-version` に `3.13` を置いて固定した
  (lock にある全パッケージが cp312 / cp313 の Windows wheel を持っている)
- `scripts/npu.ps1` の `setup` / `check` / `test` / `decide` / `bench` / `hf` はそのまま動いた。
  `check` は CPU / GPU / NPU の 3 つを表示し、`test` は 59 passed, 3 skipped (skip は torch が要る 3 つ)
- トークナイザ (`tokenizer.json`、sentencepiece) は Windows でもそのまま読めた
- コンパイル結果のキャッシュ (`.cache/openvino`) は効いている。L=128 の初回は 35.2 秒かかり、
  キャッシュがある状態の起動は 4.5 秒だった。L=512 の初回コンパイルは 65 秒

### 4.2 系列長 160〜480 で NPU がハングする

```
RuntimeError: Exception from src\inference\src\cpp\infer_request.cpp:224:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:382:
L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST, code 0x70000001
 - device hung, reset, was removed, or driver update occurred
```

長さを変えて IR を直接動かした結果 (1 長さにつき 1 プロセス。ハングすると以降が全滅するため分けた):

| 長さ | 96 | 128 | 160 | 176 | 192 | 224 | 256 | 288 | 320 | 384 | 448 | 480 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NPU | OK | OK | 不安定 | × | × | × | × | × | × | × | × | × | OK |

- **コンパイルは成功し、推論の実行時にハングする**
- 同じ IR を OpenVINO の CPU と内蔵 GPU で動かすと、どの長さでも正常 → IR ではなく NPU 側
  (ドライバ / NPU コンパイラ) の問題
- ダミー入力 (`input_ids` を全部 1、他を 0) で IR を直接叩いても再現する →
  `packing.py` や質問の内容とは無関係。マーカーの数にも相関しない
  (markers=3 でも 10 でも落ち、L=512 の markers=20 は通る)
- `.cache/openvino` を消しても再現する → キャッシュの破損ではない
- 一度ハングすると、同じプロセス内の以降の推論もコンパイルも失敗する。復帰にはプロセスの再起動が要る
- L=128 は 3 プロセス 9/9 で成功、L=512 は 4/4 で成功。L=160 は非決定的
  (成功する回、3 回目の推論で落ちる回、プロセスごと落ちる回があった)

**回避策**: `s1.json` の `lengths` から 256 を外して `[128, 512]` にすると、256 に落ちていた入力が
512 に丸められて正常に動く。この状態で `noul-3` と `noul-10` の答えが OpenVINO の CPU と一致した
(noul の確率の差は最大 0.0055。FP16 相当の誤差)。

### 4.3 速さ (p50、warmup 3 / runs 20)

NPU は `lengths` を `[128, 512]` にしたモデルディレクトリ、CPU と GPU は `[128, 256, 512]` のまま。
そのため `noul-3` と `noul-10` は NPU だけ L=512 で、他は L=256 で測っている。

| 入力 | 長さ (NPU) | NPU | 内蔵 GPU | CPU |
|---|---|---|---|---|
| `examples/review.yaml` (noul 2 + choice 3 択) | 128 | 708 ms | | |
| `bench/noul-1.yaml` | 128 | 769 ms | | |
| `bench/noul-3.yaml` | 512 | 803 ms | 126 ms | 1,371 ms |
| `bench/noul-10.yaml` | 512 | 835 ms | 125 ms | 1,261 ms |
| `bench/choice-20.yaml` | 512 | 1,027 ms | | |

- **内蔵 GPU (Arc Graphics) が NPU より 6〜7 倍速い**。プラン フェーズ 0-b の判定
  「NPU で動かない、または極端に遅い場合は、NPU PC の内蔵 GPU で動かす」に該当する
- ただし NPU 側は何も調整していない (`PERFORMANCE_HINT` は `LATENCY` のみ、INT8 量子化は未実施)。
  フェーズ 5 の量子化で差が縮むかは未確認
- この PC の OpenVINO の CPU は、開発 PC の CPU (2 節: 77〜392 ms) より遅い

### 4.4 次に確かめること

- ペア方式の IR (`Coxless/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-fp16-ov`、8×256) でも
  256 で落ちるか。落ちるなら mDeBERTa 全般、落ちないならパック方式のグラフ固有と分かる
- NPU ドライバを更新して再現するか (32.0.100.5540 は 2026-08-20 版)
- `lengths` の既定を変えるか、実行時に上書きできるようにするか
  (いまは開発 PC の `export` 時に `s1.json` に焼かれる)
- 内蔵 GPU を既定にするかの判定。するなら `scripts/npu.ps1` と `.workshop/npu.yaml` の
  `--device NPU` を変える

## 残っていること

1. (済み) NPU での確認と IR の Hugging Face Hub への受け渡しは 4 節。
   系列長 160〜480 で NPU がハングする問題が残っている (4.2、4.4)
2. 選択肢が多くて複数のパックに分かれる場合の速さは測っていない (`max_markers` は 32)
3. OpenVINO の CPU の速さは、スレッド数などの設定を変えていない (`PERFORMANCE_HINT` は `LATENCY`)
