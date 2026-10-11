# フェーズ 4 の結果 (エクスポートと OpenVINO 推論、開発 PC 分)

実施: 2026-10-11。環境: 開発 PC (ワークショップ `gpu`、GTX 1660 Ti、Core i7-10700F)。
NPU PC がまだないので、開発 PC でできる範囲 (torch の GPU と OpenVINO の CPU) だけを行った。
NPU での確認は残っている。

## まとめ

- パック方式のモデルを OpenVINO IR にできるようにした (`export`。方式は `s1.json` で決まる)
- OpenVINO でパック方式を推論できるようにした (`OpenVINOPackedScorer`)。系列長 128 / 256 / 512 ごとに
  静的な形状でコンパイルし、パックの長さに合うものを使う
- **torch と OpenVINO の CPU で結果が一致した**。評価セットの正解率は noul 0.934、choice 0.839 (torch は 0.934 / 0.838)
- OpenVINO の CPU での速さは、パック方式が 77〜392 ms、ペア方式が 1.7〜5.1 秒 (2 節)
- IR を静的な形状にすると、形状が動的な演算も、条件で枝が変わる `If` も残らない (3 節)。
  NPU でコンパイルできるかは、NPU PC で確かめるまで分からない

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

## 残っていること

1. **NPU での確認** (NPU PC が要る): コンパイルできるか、出力が OpenVINO の CPU と合うか、速さ
2. **IR を Hugging Face Hub に上げる**: `workshop run gpu -- hf upload Coxless/local-decision-model models/packed-ov packed-ov --private`
   (`hf auth login` が先に要る)
3. 選択肢が多くて複数のパックに分かれる場合の速さは測っていない (`max_markers` は 32)
4. OpenVINO の CPU の速さは、スレッド数などの設定を変えていない (`PERFORMANCE_HINT` は `LATENCY`)
