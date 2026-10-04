"""フェーズ 2: 2 つの教師 (ペア方式の mDeBERTa と Claude Code) のラベルを、質問ごとに比べる。

どちらも distill と同じ形式の JSONL。同じ状態の行どうしを比べる。
正解との比較ではないので、どちらが正しいかはわからない。食い違いの大きい質問を見つけて、
ラベルの混ぜ方を決める材料にする。

    workshop exec gpu -- bash -c 'cd /project && uv run python scripts/phase2/compare_teachers.py'
"""

from __future__ import annotations

import argparse

import numpy as np

from local_decision_model.distill import load_examples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="data/train-pair.jsonl")
    p.add_argument("--claude", default="data/train-claude.jsonl")
    args = p.parse_args()

    pair = {e.state: e for e in load_examples(args.pair)}
    claude = {e.state: e for e in load_examples(args.claude)}
    states = [s for s in pair if s in claude]
    print(f"ペア方式 {len(pair)} 行 / Claude {len(claude)} 行 / 両方にある状態 {len(states)} 行\n")

    names = list(pair[states[0]].labels)
    # 列: 平均の確率 (ペア方式 / Claude)、0.5 を境にした判定が同じ割合、確率の差の平均、相関
    print(f"{'noul':<14} {'ペア':>9} {'Claude':>11} {'同じ判定':>8} {'|Δp|':>9} {'相関':>6}")
    rows = []
    for name in names:
        if isinstance(pair[states[0]].labels[name], dict):
            continue
        a = np.array([pair[s].labels[name] for s in states])
        b = np.array([claude[s].labels[name] for s in states])
        rows.append((a.mean(), b.mean(), np.mean((a > 0.5) == (b > 0.5)), np.abs(a - b).mean()))
        m = rows[-1]
        corr = np.corrcoef(a, b)[0, 1]
        print(f"{name:<14} {m[0]:>9.3f} {m[1]:>11.3f} {m[2]:>8.3f} {m[3]:>9.3f} {corr:>6.3f}")
    m = np.array(rows).mean(0)
    print(f"{'(全体)':<14} {m[0]:>9.3f} {m[1]:>11.3f} {m[2]:>8.3f} {m[3]:>9.3f}\n")

    print(f"{'choice':<14} {'同じ選択':>8}  最上位の選択肢の分布 (ペア / Claude)")
    for name in names:
        first = pair[states[0]].labels[name]
        if not isinstance(first, dict):
            continue
        options = list(first)
        a = np.array([[pair[s].labels[name][o] for o in options] for s in states]).argmax(1)
        b = np.array([[claude[s].labels[name][o] for o in options] for s in states]).argmax(1)
        share = "  ".join(
            f"{o} {np.mean(a == i):.2f}/{np.mean(b == i):.2f}" for i, o in enumerate(options)
        )
        print(f"{name:<14} {np.mean(a == b):>8.3f}  {share}")


if __name__ == "__main__":
    main()
