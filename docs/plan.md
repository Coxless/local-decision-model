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
  ペア方式は比較用として残す (教師には使わない。フェーズ 2)。

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

- Git のリモートは `github.com/Coxless/local-decision-model` (現在は public)。
- `models/` は `.gitignore` 済み。NPU PC への転送は Hugging Face Hub の private リポジトリで行う
  (`hf` action。手順はフェーズ 0-b)。IR は数百 MB なので Git には入れない。

### Windows の NPU PC でワークショップの代わりにどうするか

ワークショップは Ubuntu の LXD の上で動くので Windows では使えない。WSL2 の中でも NPU は見えない
(Intel の NPU ドライバは Windows ネイティブ用で、WSL2 にはデバイスが渡らない)。そのため、
**Windows ではワークショップを使わず、uv で直接動かす**。NPU PC では推論しかしないので
(`--extra openvino` のみで、torch も CUDA も要らない)、ワークショップのような隔離がなくても困らない。

- 前提: Intel NPU ドライバ (デバイスマネージャーに「Intel(R) AI Boost」が出ていること)、Git、uv
  (`winget install astral-sh.uv`)。Python は uv が `requires-python` に合わせて自動で入れる。
- `npu.yaml` の actions と同じことをする `scripts/npu.ps1` を作る。
  使い方: `powershell -ExecutionPolicy Bypass -File scripts\npu.ps1 bench --device NPU`
  - actions は `setup` / `check` / `test` / `decide` / `bench` / `hf` (`npu.yaml` と同じ名前と引数)
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
[CLS] 状態… | [Q] 質問1 | [Q] 質問2 | choice の指示 [O] 選択肢A [O] 選択肢B | PAD
  q=0          q=1         q=2         q=3          q=3, o=1      q=3, o=2

アテンション: 状態 → 状態のみ / 質問 → 状態 + 自分の質問 / 選択肢 → 状態 + 親の質問 + 自分
相対位置:     各質問の位置番号を「状態のすぐ後ろ」から振り直す (マーカー [Q] / [O] は位置 0)
出力:         [Q] と [O] の位置の隠れ状態 → NLI ヘッド → 判断ロジット z
              noul: sigmoid(z / T_noul)    choice: softmax(z / T_choice)
```

- **質問同士が干渉しない**: 位置番号の振り直しとマスクにより、ある質問の答えは他の質問の有無や
  順番に関係なく同じになる。テストで保証する。
- **ヘッドを流用できる**: NLI 学習済みの分類ヘッド (`entailment - contradiction`) を初期値に使い、
  学習前でもある程度動く状態から始める。
- **分かっている弱点**: ペア方式では状態のトークンも仮説を見ていたが、この方式では見ない。
  精度が少し下がるはずなので、蒸留で取り戻す。フェーズ 3 で「状態も質問を見る」版と比較する。

## フェーズ 0: 下調べ (NPU で動くかを先に確認する)

一番のリスクは「mDeBERTa が NPU でコンパイルできるか、十分速いか」なので最初に確かめる。

**0-a. 開発 PC (Ubuntu, ワークショップ `gpu`)** — 完了。結果は [phase0-results.md](phase0-results.md)

- [x] 既存の CLI で mDeBERTa の `decide`、`bench` を GPU で試し、`export` で IR を作る (8×256 と 1×512)
- [x] `config.json` の `position_biased_input` が false か確認する
      (false なら絶対位置の埋め込みがなく、位置の振り直しだけで質問同士の独立が成り立つ)
- [x] 6 GB の GPU で fp32 学習が回るか、小さな学習ループでピークメモリと速さを測る
      (単語埋め込みの凍結 + gradient checkpointing。見積もりは約 3〜4 GB)
- [x] (比較) 既存の 1 パス分類モデル GLiClass の多言語版を日本語の例題で試す
- [x] TypeSafe API との違いを洗い出す (フェーズ 6 の「互換にする範囲」を参照)。
      特に、jev の `instructions` は疑問文 (「〜ですか?」) で、NLI は平叙文の仮説を前提にしている点。
      疑問文のまま zero-shot でどれだけ精度が落ちるかを測る
- [x] Git のリモートを用意して push する。`.gitattributes` と `scripts/npu.ps1` もここで作る

**0-b. NPU PC (Windows, uv をそのまま使う)**

モデルは Hugging Face Hub の private リポジトリ
`Coxless/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-fp16-ov` 経由で渡す
(0-a で作った `models/mdeberta-ov`。ベースモデルを IR に変換しただけで、学習した重みは入っていない)。
`scripts\npu.ps1` は `powershell -ExecutionPolicy Bypass -File scripts\npu.ps1 <action>` で実行する。

- [ ] (開発 PC) IR を上げる:
      `workshop run gpu -- hf auth login` →
      `workshop run gpu -- hf upload Coxless/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-fp16-ov models/mdeberta-ov . --private`
- [ ] `git clone` → `scripts\npu.ps1 setup` → `check` で、OpenVINO のデバイス一覧に NPU が出るか確認する
- [ ] IR を取得する: `scripts\npu.ps1 hf auth login` (read 権限のトークン) →
      `scripts\npu.ps1 hf download Coxless/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-fp16-ov --local-dir models\mdeberta-ov`
- [ ] 1×512 用のモデルディレクトリを作る: `models\mdeberta-ov` を `models\mdeberta-ov-512` にコピーし、
      `s1.json` の `max_length` を 512 に書き換える
- [ ] `decide` と `bench` を NPU / GPU / CPU で比べる:
      `scripts\npu.ps1 decide --model models\mdeberta-ov`、
      `scripts\npu.ps1 bench --model models\mdeberta-ov --device NPU`、
      `scripts\npu.ps1 bench --model models\mdeberta-ov-512 --batch-size 1 --device NPU`
      (基準は開発 PC の OpenVINO CPU の p50: 8×256 が 1,253 ms、1×512 が 1,841 ms)
- [ ] トークナイザ (sentencepiece) が Windows でも読めるか確認する
- [ ] NPU のコンパイルにかかる時間と、`.cache/openvino` のキャッシュが効くかを確認する

**判定**: NPU で動かない、または極端に遅い場合は、NPU PC の内蔵 GPU で動かすか、
ベースを XLM-R 系 (MiniLM) に変える。後者は位置の振り直しが DeBERTa 固有なので作り直しが必要。

## フェーズ 1: パック入力とモデル

完了。結果とプランから変えた点は [phase1-results.md](phase1-results.md)。

| ファイル | 内容 |
|---|---|
| `src/local_decision_model/packing.py` (新規) | 状態と質問から `input_ids`、`question_ids`、`option_ids`、`positions`、`marker_index` と、マーカー → (質問, 選択肢) の対応表を作る。numpy だけで書く。長すぎるときは状態を切り詰め、マーカーが入りきらないときは複数回の推論に分ける |
| `src/local_decision_model/packed_model.py` (新規) | `PackedModel(nn.Module)`。`question_ids`、`option_ids`、`positions` からグラフ内でマスクと相対位置を作り、`embeddings` → `encoder` → マーカー位置の取り出し → NLI ヘッド、の順に通す |
| `scoring.py` | `InferenceConfig` に `arch: "pair" \| "packed"`、`temperature_noul`、`temperature_choice`、`lengths: [128, 256, 512]`、`max_markers`、`max_state_tokens`、`marker_position`、`choice_layout` を追加。`DEFAULT_BASE_MODEL` を mDeBERTa に変更 |
| `decide.py` | `Decider` に、パック方式のバックエンド (`decide(state, questions)` で直接答えを返す) の経路を追加 |

- マーカーは新しい語彙を足さず、既存のトークン (`[CLS]` など) に**学習可能な役割埋め込み**
  (状態 / 質問 / 選択肢) を足して区別する。単語埋め込みを凍結したまま学習できる。
- マスクと相対位置をグラフ内で作るので、NPU に渡す入力は `[1, L]` の整数テンソル 4 個と `marker_index` で済む。
- DeBERTa は相対位置を対数のバケットに変換して使うので、自分で `relative_pos` を渡すときも
  その変換を通す。

**テスト** (小さなランダム初期化の DeBERTa 設定を使い、CPU で実行):

- 質問を 1 つだけ渡したときと、他の質問と一緒に渡したときで答えが一致する
- 質問や選択肢の順番を入れ替えても答えが変わらない
- 状態の切り詰め、推論の分割、パディングの扱いが正しい

**確認**: 学習前の zero-shot の状態で、ペア方式とどれだけ答えが一致するかを測る
(フェーズ 3 で比べる基準)。

## フェーズ 2: データ

データ作りまで完了。結果は [phase2-results.md](phase2-results.md)。評価セットの人による見直しが残っている。

教師は **Claude Code (`claude -p`) だけ**にする (2026-10-05 に決定)。ペア方式の mDeBERTa は教師に使わない。

- 経緯: はじめはペア方式の mDeBERTa を教師にし、大きな LLM のラベルを混ぜる計画だった。
  実際に比べると、zero-shot のペア方式はほとんどの質問に「真」と答え、Claude と noul の判定が 37% しか合わなかった。
  Sonnet と Opus の判定は 95% 一致した (フェーズ 2 の結果の 3 節)
- 学習データは `data/train-claude.jsonl` (`scripts/phase2/label_with_claude.py` が、確率を 0〜100 の整数で答えさせて書く)
- ペア方式は教師ではなく、パック方式と比べる基準として残す。`distill` はペア方式のモデルでラベルを付けるコマンドで、
  比較用に使う (`data/train-pair.jsonl` は zero-shot のペア方式のラベル 704 件。学習には使わない)
- ラベルどうしの比較は `scripts/phase2/compare_teachers.py`

- [x] `src/local_decision_model/distill.py` (新規): 状態テキストと質問セットを入力に、既存の `TorchScorer` で
      ペア方式の確率を付ける
- [x] データ形式を「1 行 = 1 つの状態 + 複数の質問」に変える。質問の書き方は `examples/*.yaml` と同じ
      (`distill.load_examples` で読む。ペア方式の `train` は 1 行 1 ペアの形式のままで、フェーズ 3 で移す)

```json
{"state": "…", "questions": {"refund": "顧客は返金を求めている。", "topic": {"type": "choice", "instructions": "…{option}…", "options": ["配送", "品質", "その他"]}},
 "labels": {"refund": 0.94, "topic": {"配送": 0.81, "品質": 0.12, "その他": 0.07}}}
```

- **状態テキスト**: Claude Code に書かせる (`scripts/phase2/generate_states.py` → `data/states.jsonl`)。
  業種、用件、感情、長さ、文体などの条件をスクリプトが乱数で決め、条件ごとに 1 件ずつ書かせる。
- **質問セット**: `data/questions.yaml` (noul 36 問 + choice 5 問)。noul は疑問文で書き、一部は平叙文の言い換えも入れる。
  ラベルは教師が自動で付けるので、用意するのは質問だけ。
- **評価セット**: 人が正解を付けたものを数百件、学習用とは別に用意する (状態は `data/eval-states.jsonl` に 300 件を生成済み)。教師との一致だけでは
  教師の間違いを真似していても気付けないので、較正の評価はこちらで行う。

## フェーズ 3: 学習と較正

完了。結果とプランから変えた点は [phase3-results.md](phase3-results.md)。合格ラインを満たし、「状態は質問を見ない」版を採用した。

- [x] `train.py` にパック方式の学習を追加する
  - noul はソフトラベルの BCE、choice は教師の分布とのクロスエントロピー
  - 単語埋め込みは凍結、fp32、gradient checkpointing、小さいバッチを勾配の累積で補う
  - 毎ステップ、質問の数 (1〜K)、質問の順番、選択肢の順番をランダムに変える
- [x] 較正: noul と choice で別々に温度を求めて `s1.json` に保存する。
      評価指標は正解率、NLL、ECE (choice は最上位の答えの ECE)。
      温度は、ラベルを 0 / 1 に丸めてから求める (教師の確率のままだと ECE が悪くなった)
- [x] 比較実験: 「状態は質問を見ない」版 (キャッシュできて独立) と「状態も質問を見る」版
      (精度は上がるかもしれないが質問同士が干渉する) を比べる。差が小さければ前者を採用する

**合格ライン**: 評価セットで、同じ学習データで学習したペア方式との正解率の差が 1〜2 ポイント以内、
ECE はペア方式と同等以下。結果は noul が −0.3 ポイント、choice が +0.5 ポイントで、ECE も同じ水準。

## フェーズ 4: エクスポートと NPU 推論

- [ ] `export.py`: パック方式のモデルを OpenVINO IR に変換する
      (入力は `input_ids`、`question_ids`、`option_ids`、`positions`、`marker_index`、出力は `logits[1, M]`)
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
      (`train --arch packed` と `evaluate` はフェーズ 3、`distill` はフェーズ 2 で追加済み)。
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
| `POST /v1/systemone`、本文は `state` / `model` / `questions` | 対応する。`model` は `local-decision-model-…` のような独自の名前にし、`jev-latest` などは既定のモデルの別名として受け付ける |
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

型名は `noul` にそろえた (以前の `bool` も読める)。回答の形と `{option}` テンプレートは、
上の形 (`{"type": "noul", "noul": p}`、`criteria`) に寄せる。
古い形式の YAML (`examples/*.yaml`) は、読み込み時に変換して受け付ける。

- [ ] `src/local_decision_model/server.py` (新規): 標準ライブラリか FastAPI の薄いサーバー。起動時にモデルを読み込んで
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

1. **対象の分野**: 問い合わせ (顧客からの問い合わせ文) にする (2026-10-03 に決定)。
2. **フェーズ 0 の結果**: NPU での速さは開発 PC では測れない。NPU PC が手元にないため、0-b は後回しにして
   フェーズ 1 を先に進めた。NPU で動かない場合は `packed_model.py` を作り直す。
3. **モデルの転送方法**: Hugging Face Hub の private リポジトリにする (2026-10-06 に決定)。
   ベースモデルを変換しただけの IR は `Coxless/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-fp16-ov`、
   自分で学習したモデルは別のリポジトリ `Coxless/local-decision-model` に分け、公開するかどうかを別々に決められるようにする。
4. **Windows の NPU ドライバのバージョン**: OpenVINO の NPU プラグインは、ドライバ側のコンパイラに
   依存する。Windows のドライバのバージョンを記録し、動かなければ更新する。
5. **サーバーのポート番号と起動方法**: Ollama (11434) などとぶつからない番号にする。
   常駐の方法 (手動起動、スタートアップ、Windows サービス) は、使うアプリが決まってから決める。
6. **疑問文の instructions への対応**: 学習データの質問を jev と同じ疑問文で書いて学習させるか、
   疑問文を平叙文に変換するか。フェーズ 0 の測定結果を見て決める。
7. **フェーズ 3 の合格ラインの基準**: 同じ学習データ (`data/train-claude.jsonl`) でファインチューニングしたペア方式を
   基準にした (2026-10-05 に決定)。
8. **学習にない質問での精度**: フェーズ 3 の評価は学習と同じ 41 問で測った。質問を自由に書いたときにどれだけ当たるかは未確認。
   質問セットを増やすか、学習に使わない質問を評価用に取り分けるかを決める。
