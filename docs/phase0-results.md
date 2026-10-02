# フェーズ 0-a の結果 (開発 PC)

実施: 2026-10-03。環境: ワークショップ `gpu` (GTX 1660 Ti 6 GB, i7-10700F)、
torch 2.11.0+cu128、transformers 5.18.0、openvino 2026.4.1。

モデル: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (以下 mDeBERTa)、
比較に既存の `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (以下 MiniLM)。

## まとめ

- mDeBERTa は既存の CLI のまま `decide` / `bench` / `export` が通る。OpenVINO CPU の出力は torch と小数第 4 位まで一致
- **transformers 5 は既定で config の dtype (mDeBERTa は float16) で読み込む**。fp16 だと GTX 1660 Ti では
  3.7 倍遅かった (447 ms → fp32 で 120 ms)。`torch_backend` / `export` / `train` で `dtype=torch.float32` を明示した
- `position_biased_input` は **false**。絶対位置の埋め込みがないので、位置の振り直しだけで質問同士を独立にできる
- 6 GB の GPU での fp32 学習は **ピーク約 2.4 GiB** で余裕あり (見積もり 3〜4 GB より小さい)。NaN も出なかった
- 疑問文の instructions でも zero-shot の精度は落ちなかった (ただし 60 問で差はノイズの範囲)
- **jev 形式の choice (instructions に `{option}` がなく、選択肢を後ろにつなげる) は大きく落ちる** (0.75 → 0.40)。
  選択肢ごとの仮説文の作り方がフェーズ 6 の互換対応で一番の課題になる
- GLiClass `multilang-mini` は topic の正解率が mDeBERTa と同じ 0.75 で、1 件 9.5 ms (1 パス)

## 1. decide / bench / export

`examples/review.yaml` (bool 2 問 + choice 3 択 = 5 ペア、長さ 256)。

| モデル | バックエンド | p50 |
|---|---|---|
| MiniLM | torch / cuda (fp32) | 13 ms |
| mDeBERTa | torch / cuda (fp16、修正前) | 447 ms |
| mDeBERTa | torch / cuda (fp32) | 121 ms |
| mDeBERTa | torch / cpu (fp16、修正前) | 7,537 ms |
| mDeBERTa | openvino / CPU, 8×256 | 1,253 ms |
| mDeBERTa | openvino / CPU, 1×512 (5 回に分けて推論) | 1,841 ms |

- 回答はどれも妥当 (`positive` 0.9997、`recommends` 0.991、`genre` = a novel 0.985)
- IR は `models/mdeberta-ov` (FP16 圧縮で 566 MB、うち単語埋め込みが大半)。
  IR は動的形状で保存し、読み込み時に `[--batch-size, s1.json の max_length]` に reshape する。
  1×512 用は `s1.json` の `max_length` だけ 512 にした `models/mdeberta-ov-512` (重みはハードリンク)
- `tokenizer.json` (fast トークナイザ) で保存されるので、NPU PC では sentencepiece のモデルファイルは要らないはず (0-b で確認)

## 2. config.json

`position_biased_input: false`、`relative_attention: true`、`position_buckets: 256`、
`max_relative_positions: -1`、`pos_att_type: [p2c, c2p]`、`share_att_key: true`、`torch_dtype: float16`。

## 3. fp32 学習のメモリと速さ

`scripts/phase0/train_memory.py`。単語埋め込みを凍結 (279M 中 学習するのは 86M)、
gradient checkpointing、AdamW。

| バッチ × 長さ | ピーク GiB | ms/step | tokens/s |
|---|---|---|---|
| 8 × 128 | 2.38 | 378 | 2,707 |
| 16 × 128 | 2.37 | 627 | 3,264 |
| 4 × 256 | 2.38 | 402 | 2,546 |
| 8 × 256 | 2.37 | 675 | 3,036 |
| 2 × 512 | 2.38 | 501 | 2,045 |
| 4 × 512 | 2.41 | 815 | 2,514 |

- ピークは重み + Adam の状態 (約 1 GiB) が大半で、活性化は gradient checkpointing でほぼ一定
- 約 3,000 tokens/s。パック方式で 1 例 256 トークンなら 1 万例 × 3 エポックで約 45 分

## 4. 日本語の zero-shot (疑問文 vs 平叙文)

`scripts/phase0/zero_shot_ja.py`、評価セットは `scripts/phase0/ja_eval.yaml`
(問い合わせ文 20 件に人手でラベル付け。noul 60 問 (真 30)、topic 5 択 20 問)。
規模が小さいので、±6 ポイント程度の差はノイズとみなす。温度は 1.0 (較正なし)。

| モデル | noul 平叙文 | noul 疑問文 | topic 平叙文 | topic 疑問文 | topic 疑問文+選択肢 |
|---|---|---|---|---|---|
| mDeBERTa | 0.667 (ECE 0.27) | 0.717 (ECE 0.20) | 0.750 | 0.750 | **0.400** |
| MiniLM | 0.600 (ECE 0.33) | 0.533 (ECE 0.33) | 0.300 | 0.300 | 0.300 |

- 平叙文: 「顧客は返金を求めている。」/ 疑問文: 「顧客は返金を求めていますか？」
- topic の仮説文: 平叙文「この問い合わせは{option}に関するものだ。」/
  疑問文「この問い合わせは{option}に関するものですか？」/
  疑問文+選択肢「この問い合わせは何に関するものですか？ {option}」(jev の choice をそのまま今の `Choice` に渡した形)
- mDeBERTa は疑問文でも落ちない。どちらの形でも「真」に寄る傾向 (真が 50% なのに平均 P(真) が 0.70〜0.75) があり、
  較正していない状態では ECE が大きい
- MiniLM は日本語の topic でほぼ当てずっぽう。ベースを mDeBERTa に変える判断を裏付ける

**未決事項 6 への示唆**: noul は疑問文のままでよさそう (学習データも jev と同じ疑問文で書けばよい)。
choice は「疑問文 + 選択肢」の連結では NLI として成り立たないので、選択肢ごとの平叙文の仮説を作る仕組み
(テンプレート、`criteria` の説明文の利用、または疑問文+選択肢の形で学習させる) が必要。

## 5. GLiClass との比較

`scripts/phase0/gliclass_ja.py` (`uv run --with gliclass` で一時的に追加。プロジェクトの依存には入れていない)。

| モデル | topic (5 択) | 速さ | noul (ラベル 1 個の多ラベル分類) |
|---|---|---|---|
| `knowledgator/gliclass-multilang-mini` | 0.750 | 9.5 ms/件 | 0.717 |
| `knowledgator/gliclass-x-base` (mDeBERTa ベース) | 0.750 | 8.5 ms/件 | 0.500 |
| `knowledgator/gliclass-multilang-edge` | 動かず (triton が C コンパイラを要求。ワークショップに gcc がない) | | |

- topic の正解率は mDeBERTa のペア方式と同じで、1 パスなので速い。パック方式の目標の参考になる
- noul を「ラベル 1 個」で使うのは x-base では成り立たない (しきい値 0.5 で全部同じ側)。
  GLiClass はラベル同士の比較が前提で、単独の真偽の確率としては使いにくい
- `gliclass` を入れると torch が 2.14.1+cu130 (PyPI) に入れ替わる。プロジェクトの環境 (cu128) には混ぜない

## 6. TypeSafe API との違い

[API リファレンス](https://docs.typesafe.ai/api.md)、[Models](https://docs.typesafe.ai/models.md)、
[Confidence](https://docs.typesafe.ai/confidence.md) (2026-10-03 時点) と今のコードの比較。

| 項目 | TypeSafe | 今のコード | 対応の難しさ |
|---|---|---|---|
| 質問の型名 | `noul` / `choice` / `score` | `bool` / `choice` | 小。名前の変更と旧形式の変換 |
| noul の instructions | 疑問文 (「Does this convey urgency?」) | 平叙文の仮説 | 小。zero-shot でも疑問文で落ちない (4 節) |
| noul の `criteria.true` / `false` | 任意 | なし | 中。`true` を仮説文に使う、または状態に足す |
| choice の選択肢 | `criteria` のキー (選択肢 → 説明文 or null)、最大 255 | `options` の配列 + `{option}` テンプレート | **大**。テンプレートがないので仮説文の作り方を決める必要がある (4 節) |
| score | 2〜10 段階の順序付き水準 | なし | 中。choice として確率を出し、加重平均と `legend` を付ける |
| instructions / criteria がオブジェクト・配列 | 可 (バッククォートで state のフィールドを参照) | 文字列のみ | 中。まずは JSON 文字列にして連結 |
| state がオブジェクト・配列 | 可 (推奨はオブジェクト) | 文字列のみ | 中。JSON 文字列化で精度がどうなるか要検証 |
| 回答 (noul) | `{"type": "noul", "noul": p}` | `{"type": "bool", "value", "probability"}` | 小 |
| 回答 (choice) | `choice` / `probabilities` / `confidence` | `value` / `probabilities` | 小 |
| confidence (choice) | `(n·p_max − 1) / (n − 1)` を 0〜1 に切り詰め | なし | 小。式はドキュメントのソースで確認済み |
| confidence (score) | `1 − Σ p_i·|i − peak| / (Σ_i |i − (n−1)/2| / n)` を 0〜1 に切り詰め | なし | 小。分母は「一様分布の中心からの平均距離」(プランの「一様分布の広がり」はこの意味) |
| レスポンス | `model` (版付き ID)、`usage` | なし | 小 |
| モデル一覧 | `GET /v1/models` → `{"models": [{name, description, release_date}]}` | なし | 小 |
| エイリアス | `jev-latest`、`jev-preview` | なし | 小 |
| 長さ | state + 一番長い質問で 32k トークン | 256 / 512 トークン | 仕様上の差。長い state は切り詰めになることを明記する |
| エラー | 401 / 422 / 429 / 529 | なし | 小。422 のみ同じ形で返す |
| 言語 | 英語が主で CJK は精度が落ちると明記 | 多言語 NLI | 日本語ではこちらに利点がありうる |

## 7. Git とリモート

- リモート `origin` (`github.com/Coxless/local-decision-model`) は作成済みで push 済み。
  **ただしリポジトリは public** (プランでは private を想定)。必要なら GitHub 側で private に変更する
- `.gitattributes` (`* text=auto eol=lf`) を追加
- `scripts/npu.ps1` を追加。Windows PowerShell 5.1 は BOM なしの UTF-8 を ANSI として読むので、
  このファイルは ASCII のみで書いている (日本語のコメントを入れない)

## 0-b (NPU PC) への引き継ぎ

- 転送するもの: `models/mdeberta-ov` (566 MB) と `models/mdeberta-ov-512` (s1.json 以外は同じファイル)
- 実行例:
  `scripts\npu.ps1 decide --model models\mdeberta-ov`、
  `scripts\npu.ps1 bench --model models\mdeberta-ov --device NPU`、
  `scripts\npu.ps1 bench --model models\mdeberta-ov-512 --batch-size 1 --device NPU`
  (`--device GPU` / `CPU` で比較)
- 比較の基準: 開発 PC の OpenVINO CPU で 8×256 が p50 1,253 ms、1×512 が 1,841 ms
