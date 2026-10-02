# 計画: mDeBERTa-v3-base による 1 パス判断モデル

作成: 2026-10-03

jev の「構造化出力専用の仕組み」と「較正された確率」を取り入れ、賢さは小型モデル相当のまま、
ローカル (Intel NPU) で高速に動く判断モデルを作る。ベースは
[`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)。

## 前提 (確認済み)

- transformers 5.18 の DeBERTa-v2 は、エンコーダに 3 次元のアテンションマスク `[B, L, L]` と
  相対位置 `relative_pos` を渡せる。ただし埋め込み層 (`DebertaV2Embeddings`) は 3 次元マスクを
  受け付けないため、`embeddings` と `encoder` を別々に呼ぶ forward を自作する。
- 開発 PC の GPU は GTX 1660 Ti (6 GB)。bf16 が使えず、mDeBERTa は fp16 で NaN が出やすいので、
  学習は fp32 + メモリ節約 (単語埋め込みの凍結、gradient checkpointing、勾配の累積) で行う。
- 今のコードは `Scorer.score(pairs)` が (状態, 仮説) のペア単位。パック入力用の経路を新しく足し、
  ペア方式は教師モデル・比較用として残す。

## 実行環境と PC 間の流れ

| PC | OS | 環境 | 役割 |
|---|---|---|---|
| 開発 PC | Ubuntu | ワークショップ `gpu` (GTX 1660 Ti) | 学習、評価、OpenVINO IR へのエクスポート、GPU での動作確認 |
| NPU PC | Windows | uv をそのまま使う (ワークショップなし) | NPU での動作確認とベンチマーク |

```
開発 PC: 実装 → GPU で動作確認 → export (models/*-ov) → git push
                                       │ モデルは Git に入れず別に転送
NPU PC:  git clone / pull → setup → check → decide / bench (NPU / GPU / CPU)
```

- まだコミットもリモートもないので、最初に Git のリモート (GitHub の private リポジトリなど) を用意する。
- `models/` は `.gitignore` 済み。NPU PC への転送は `scp` (Windows 標準の OpenSSH で受けられる)、
  共有フォルダ、zip のどれかで行う。IR は数百 MB なので Git には入れない。

### Windows の NPU PC でワークショップの代わりにどうするか

ワークショップは Ubuntu の LXD の上で動くので Windows では使えない。WSL2 の中でも NPU は見えない
(Intel の NPU ドライバは Windows ネイティブ用で、WSL2 にはデバイスが渡らない)。そのため、
**Windows ではワークショップを使わず、uv で直接動かす**。NPU PC では推論しかしないので
(`--extra openvino` のみで、torch も CUDA も要らない)、ワークショップのような隔離がなくても困らない。

- 前提: Intel NPU ドライバ (デバイスマネージャーに「Intel(R) AI Boost」が出ていること)、Git、uv
  (`winget install astral-sh.uv`)。Python は uv が `requires-python` に合わせて自動で入れる。
- `npu.yaml` の actions と同じことをする `scripts/npu.ps1` を作る。
  使い方: `powershell -ExecutionPolicy Bypass -File scripts\npu.ps1 bench --device NPU`
  - actions は `setup` / `check` / `test` / `decide` / `bench` (`npu.yaml` と同じ名前と引数)
  - ワークショップの環境変数に合わせて、`HF_HOME=.cache\huggingface` をスクリプト内で設定する
  - venv はプロジェクト内の `.venv` を使う (uv の既定)
  - `check` は `/dev/accel` の代わりに、OpenVINO のデバイス一覧に `NPU` があるかを確認する
- `.workshop/npu.yaml` は Ubuntu の NPU PC 用として残す。actions を変えるときは `npu.ps1` も同時に直す。
- `.gitattributes` を追加し、改行コードを LF に統一する (Windows で clone しても `.workshop/*/hooks` の
  シェルスクリプトが壊れないようにする)。
- ソースコードはパスを `pathlib` で扱っており、OS に依存した処理はない (確認済み)。新しいコードもこれに合わせる。

## 全体像

```
入力 (1 本の系列、長さは 128 / 256 / 512 のどれかに固定)
[CLS] 状態… | [Q] 質問1 | [Q] 質問2 | [Q] choice の指示 [O] 選択肢A [O] 選択肢B | PAD
  seg=0        seg=1       seg=2       seg=3              seg=4         seg=5

アテンション: 状態 → 状態のみ / 質問 → 状態 + 自分の質問 / 選択肢 → 状態 + 親の質問 + 自分
相対位置:     各質問の位置番号を「状態のすぐ後ろ」から振り直す
出力:         [Q] と [O] の位置の隠れ状態 → NLI ヘッド → 判断ロジット z
              bool: sigmoid(z / T_bool)    choice: softmax(z / T_choice)
```

- **質問同士が干渉しない**: 位置番号の振り直しとマスクにより、ある質問の答えは他の質問の有無や
  順番に関係なく同じになる。テストで保証する。
- **ヘッドを流用できる**: NLI 学習済みの分類ヘッド (`entailment - contradiction`) を初期値に使い、
  学習前でもある程度動く状態から始める。
- **分かっている弱点**: ペア方式では状態のトークンも仮説を見ていたが、この方式では見ない。
  精度が少し下がるはずなので、蒸留で取り戻す。フェーズ 3 で「状態も質問を見る」版と比較する。

## フェーズ 0: 下調べ (NPU で動くかを先に確認する)

一番のリスクは「mDeBERTa が NPU でコンパイルできるか、十分速いか」なので最初に確かめる。

**0-a. 開発 PC (Ubuntu, ワークショップ `gpu`)**

- [ ] 既存の CLI で mDeBERTa の `decide`、`bench` を GPU で試し、`export` で IR を作る (8×256 と 1×512)
- [ ] `config.json` の `position_biased_input` が false か確認する
      (false なら絶対位置の埋め込みがなく、位置の振り直しだけで質問同士の独立が成り立つ)
- [ ] 6 GB の GPU で fp32 学習が回るか、小さな学習ループでピークメモリと速さを測る
      (単語埋め込みの凍結 + gradient checkpointing。見積もりは約 3〜4 GB)
- [ ] (比較) 既存の 1 パス分類モデル GLiClass の多言語版を日本語の例題で試す
- [ ] TypeSafe API との違いを洗い出す (フェーズ 6 の「互換にする範囲」を参照)。
      特に、jev の `instructions` は疑問文 (「〜ですか?」) で、NLI は平叙文の仮説を前提にしている点。
      疑問文のまま zero-shot でどれだけ精度が落ちるかを測る
- [ ] Git のリモートを用意して push する。`.gitattributes` と `scripts/npu.ps1` もここで作る

**0-b. NPU PC (Windows, uv をそのまま使う)**

- [ ] `git clone` → `scripts\npu.ps1 setup` → `check` で、OpenVINO のデバイス一覧に NPU が出るか確認する
- [ ] 0-a で作った IR を `models/` に転送し、`decide` と `bench` を NPU / GPU / CPU で比べる
- [ ] トークナイザ (sentencepiece) が Windows でも読めるか確認する
- [ ] NPU のコンパイルにかかる時間と、`.cache/openvino` のキャッシュが効くかを確認する

**判定**: NPU で動かない、または極端に遅い場合は、NPU PC の内蔵 GPU で動かすか、
ベースを XLM-R 系 (MiniLM) に変える。後者は位置の振り直しが DeBERTa 固有なので作り直しが必要。

## フェーズ 1: パック入力とモデル

| ファイル | 内容 |
|---|---|
| `src/local_s1/packing.py` (新規) | 状態と質問から `input_ids`、`segment_ids`、`positions`、`marker_index` と、マーカー → (質問, 選択肢) の対応表を作る。numpy だけで書く。長すぎるときは状態を切り詰め、マーカーが入りきらないときは複数回の推論に分ける |
| `src/local_s1/packed_model.py` (新規) | `PackedDecider(nn.Module)`。`segment_ids` と `positions` からグラフ内でマスクと相対位置を作り、`embeddings` → `encoder` → マーカー位置の取り出し → NLI ヘッド、の順に通す |
| `scoring.py` | `S1Config` に `arch: "pair" \| "packed"`、`temperature_bool`、`temperature_choice`、`lengths: [128, 256, 512]`、`max_markers` を追加。`DEFAULT_BASE_MODEL` を mDeBERTa に変更 |
| `decide.py` | `Decider` に、パック方式のバックエンド (`decide(state, questions)` で直接答えを返す) の経路を追加 |

- マーカーは新しい語彙を足さず、既存のトークン (`[CLS]` など) に**学習可能な役割埋め込み**
  (状態 / 質問 / 選択肢) を足して区別する。単語埋め込みを凍結したまま学習できる。
- マスクと相対位置をグラフ内で作るので、NPU に渡す入力は `[1, L]` の整数テンソル 3〜4 個で済む。
- DeBERTa は相対位置を対数のバケットに変換して使うので、自分で `relative_pos` を渡すときも
  その変換を通す。

**テスト** (小さなランダム初期化の DeBERTa 設定を使い、CPU で実行):

- 質問を 1 つだけ渡したときと、他の質問と一緒に渡したときで答えが一致する
- 質問や選択肢の順番を入れ替えても答えが変わらない
- 状態の切り詰め、推論の分割、パディングの扱いが正しい

**確認**: 学習前の zero-shot の状態で、ペア方式とどれだけ答えが一致するかを測る
(フェーズ 3 で比べる基準)。

## フェーズ 2: データ

教師は**ペア方式の mDeBERTa** にする。パック方式がペア方式の答えを 1 パスで再現するように蒸留する。
大きな LLM の教師データは、あとから同じ形式で追加できる。

- [ ] `src/local_s1/distill.py` (新規): 状態テキストと質問セットを入力に、既存の `TorchScorer` で
      教師の確率を付ける
- [ ] データ形式を「1 行 = 1 つの状態 + 複数の質問」に変える。質問の書き方は `examples/*.yaml` と同じ

```json
{"state": "…", "questions": {"refund": "顧客は返金を求めている。", "topic": {"type": "choice", "instructions": "…{option}…", "options": ["配送", "品質", "その他"]}},
 "labels": {"refund": 0.94, "topic": {"配送": 0.81, "品質": 0.12, "その他": 0.07}}}
```

- **状態テキスト**: 日本語のテキストを集める。公開コーパスを使う場合はライセンスを確認する。
- **質問セット**: 対象分野に合わせて手で数十個書くか、LLM に一度だけまとめて作らせる。
  ラベルは教師が自動で付けるので、用意するのは質問文だけ。
- **評価セット**: 人が正解を付けたものを数百件、学習用とは別に用意する。教師との一致だけでは
  教師の間違いを真似していても気付けないので、較正の評価はこちらで行う。

## フェーズ 3: 学習と較正

- [ ] `train.py` にパック方式の学習を追加する
  - bool はソフトラベルの BCE、choice は教師の分布とのクロスエントロピー
  - 単語埋め込みは凍結、fp32、gradient checkpointing、小さいバッチを勾配の累積で補う
  - 毎ステップ、質問の数 (1〜K)、質問の順番、選択肢の順番をランダムに変える
- [ ] 較正: bool と choice で別々に温度を求めて `s1.json` に保存する。
      評価指標は正解率、NLL、ECE (choice は最上位の答えの ECE)
- [ ] 比較実験: 「状態は質問を見ない」版 (キャッシュできて独立) と「状態も質問を見る」版
      (精度は上がるかもしれないが質問同士が干渉する) を比べる。差が小さければ前者を採用する

**合格ライン (目安)**: 評価セットで、ペア方式との正解率の差が 1〜2 ポイント以内、
ECE はペア方式と同等以下。

## フェーズ 4: エクスポートと NPU 推論

- [ ] `export.py`: パック方式のモデルを OpenVINO IR に変換する
      (入力は `input_ids`、`segment_ids`、`positions`、`marker_index`、出力は `logits[1, M]`)
- [ ] `openvino_backend.py`: 長さ 128 / 256 / 512 の 3 種類をコンパイルしてキャッシュし、
      入力に収まる一番短いものを選ぶ
- [ ] テスト: 同じ入力に対して torch の CPU と OpenVINO の CPU で結果が一致すること
- [ ] `bench` を拡張する: 質問 1 / 3 / 10 個、choice の選択肢 20 個の各条件で、
      ペア方式とパック方式の速さを NPU / GPU / CPU で比べる
- [ ] 開発 PC で、GPU (torch) と CPU (OpenVINO) の両方で動作確認してから IR を NPU PC に転送する
- [ ] NPU PC (Windows) で `git pull` → `scripts\npu.ps1 test` → `decide` → `bench` を行い、
      NPU の出力が開発 PC の OpenVINO CPU の出力と (許容誤差内で) 一致することを確認する

## フェーズ 5: 高速化と仕上げ

- [ ] NNCF で INT8 に量子化する。較正には学習データの一部を使い、量子化前後で ECE と正解率が
      悪化していないか確認する
- [ ] 保留の判定: 確率がしきい値未満の質問を「要確認」として返すオプションを付ける
- [ ] (任意) 型の追加: 複数選択 (選択肢ごとに sigmoid)、数値 (区間に分けた choice)
- [ ] README と `.workshop/*.yaml` の actions を更新する
      (`distill` を追加し、`train` に `--arch packed` を追加)。
      README に Windows の NPU PC の手順 (ドライバ確認、uv、`scripts/npu.ps1`、モデルの転送) を追加する

## フェーズ 6: ローカルサーバー (アプリから使えるようにする)

Ollama のように、**常駐するローカルサーバーが NPU を持ち、アプリは HTTP で問い合わせる**形にする。
API は独自に決めず、**TypeSafe (jev) の HTTP API と互換にする**。こうすると、アプリはベース URL を
変えるだけで jev のクラウド、Ollaya などの互換サーバー、このサーバーを切り替えられる。
TypeSafe の公式 SDK も、`base_url` 引数 (環境変数 `TYPESAFE_BASE_URL`) で接続先を変えられる。

**サーバーにする理由**

- NPU はコンパイルに時間がかかる。常駐させれば、読み込みとコンパイルは起動時の 1 回で済む
- どの言語のアプリからも使える。NPU を複数のアプリで共有できる (リクエストは順番に処理する)
- Python の起動の遅さ (`transformers` の読み込みで数秒) が問題にならない

**互換にする範囲** ([API リファレンス](https://docs.typesafe.ai/api.md)、2026-10-03 時点)

| TypeSafe API | 対応 |
|---|---|
| `POST /v1/systemone`、本文は `state` / `model` / `questions` | 対応する。`model` は `local-s1-…` のような独自の名前にし、`jev-latest` などは既定のモデルの別名として受け付ける |
| `state` が文字列 | そのまま使う |
| `state` がオブジェクトや配列 | JSON 文字列にしてから状態テキストとして扱う (精度は要検証) |
| `noul` (+ `criteria.true` / `false`) | 対応する。回答は `{"type": "noul", "noul": p}`。`criteria.true` があれば仮説文に使う |
| `choice` (`criteria` は選択肢 → 説明、最大 255) | 対応する。回答は `choice` / `probabilities` / `confidence`。選択肢が多いときはパック方式を複数回に分けて推論する |
| `score` (最大 10 段階の順序付きの水準) | 後から対応する。水準を choice として確率を出し、`score` = 確率で重み付けした平均、`legend` を付ける |
| `instructions` がオブジェクトや配列 | 後から対応する。まずは文字列だけ |
| `confidence` | ドキュメントの式に合わせる。choice は `(n·p_max − 1) / (n − 1)`、score は「最大の水準からの確率の広がり」を一様分布の広がりで割って 1 から引いたもの。公式の実装と同じか要確認 |
| `usage` (`input_tokens` / `output_tokens`) | `input_tokens` はトークン数、`output_tokens` は 0 を返す |
| モデル一覧 (SDK の `client.models.list()`) | 対応する。HTTP のパスは SDK のソースで要確認 |
| エラー (`422`、`429` など) | `422` (検証エラー) は同じ形で返す。認証はなし (`Authorization` ヘッダーは無視する) |

今の `schema.py` の `bool` 型と `{option}` テンプレートは、上の形 (`noul`、`criteria`) に寄せる。
古い形式の YAML (`examples/*.yaml`) は、読み込み時に変換して受け付ける。

- [ ] `src/local_s1/server.py` (新規): 標準ライブラリか FastAPI の薄いサーバー。起動時にモデルを読み込んで
      NPU でコンパイルし、リクエストは 1 本のワーカーで順番に処理する。`decide.py` をそのまま呼ぶ
- [ ] CLI に `serve` を追加する (`--host 127.0.0.1 --port <番号> --model models/... --device NPU`)。
      既定は localhost のみで待ち受け、外部には公開しない
- [ ] `/health` (状態確認) を足す。起動直後のコンパイル中は「準備中」を返す
- [ ] テスト: TypeSafe のドキュメントにあるリクエスト例をそのまま送り、同じ形の回答が返ること。
      公式の Python SDK で `base_url` をこのサーバーに向けて呼べること
- [ ] Windows: `scripts\npu.ps1 serve` で起動する。必要になったら、スタートアップへの登録や
      Windows サービス化を検討する。`.workshop/*.yaml` にも `serve` の action を足す
- [ ] (比較) Ollaya など TypeSafe 互換のローカルサーバーで小型の LLM 判断モデルを動かし、同じリクエストで
      精度と速さを比べる
- [ ] (任意) 配布しやすくしたくなったら、サーバーを Rust (OpenVINO の Rust バインディング + `tokenizers`)
      の exe に置き換える。API が同じなのでアプリ側は変えなくてよい

## 未決事項

1. **対象の分野**: 状態テキスト・質問セット・評価セットを作るため、どんな文章 (問い合わせ、ログ、
   レビューなど) に使うかを決める。未定なら汎用の日本語テキストと例題の質問で始める。
2. **フェーズ 0 の結果**: NPU での速さは開発 PC では測れないので、NPU PC (Windows) の `bench` 結果を
   見てからフェーズ 1 に進む。
3. **モデルの転送方法**: `scp`、共有フォルダ、zip のどれにするか。何度も転送するようなら、
   Hugging Face Hub の private リポジトリに置いて NPU PC から取得する方法も検討する。
4. **Windows の NPU ドライバのバージョン**: OpenVINO の NPU プラグインは、ドライバ側のコンパイラに
   依存する。Windows のドライバのバージョンを記録し、動かなければ更新する。
5. **サーバーのポート番号と起動方法**: Ollama (11434) などとぶつからない番号にする。
   常駐の方法 (手動起動、スタートアップ、Windows サービス) は、使うアプリが決まってから決める。
6. **疑問文の instructions への対応**: 学習データの質問を jev と同じ疑問文で書いて学習させるか、
   疑問文を平叙文に変換するか。フェーズ 0 の測定結果を見て決める。
