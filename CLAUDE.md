# local-decision-model

「自由テキストの状態 + 名前付きの質問 → 型付き・較正された確率付きの回答」を返す小さな判断モデルを、
ローカル (CUDA / Intel NPU) で動かす。全体像は `README.md`、作業の計画は `docs/plan.md`。

## オントロジー

概念と用語の唯一の正は `docs/ontology.yaml`。

@docs/ontology.yaml

- コード、コメント、ドキュメントを書く前に、扱う概念をここで確認する。
- 新しく書く文章は `ja`、識別子は `en` の表記に合わせる。`aliases` は既存の別表記なので、新しくは書かない。
- 概念を足す、名前を変える、`code` / `files` に書かれた場所を移すときは、同じ変更の中で `ontology.yaml` も直す。
- プランにあるが未実装の概念は `status: planned` を付け、実装したら外して `code` を足す。
- `docs/phase*-results.md` は過去の記録なので、古い表記のまま残す。

## コマンド (開発 PC)

```bash
workshop run gpu -- test     # pytest
workshop run gpu -- lint     # ruff check + format --check
workshop run gpu -- fmt
```

## 書き方

- ドキュメント、コメント、docstring、コミットメッセージは日本語。
- `.workshop/*.yaml` の actions を変えるときは `scripts/npu.ps1` も同時に直す。
