"""フェーズ 2: 学習用の状態 (問い合わせ文) を Claude Code に書かせる。

条件 (業種、用件、感情、求めること、長さ、文体) の組み合わせを乱数で作り、1 回の呼び出しで
--batch 件ずつ書かせる。似た文ばかりにならないよう、条件はスクリプトの側で決める。

出力は distill にそのまま渡せる JSONL。meta は書かせたときの条件で、正解ラベルではない
(条件どおりに書かれているとは限らない)。分野の偏りを見るときにだけ使う。

    {"state": "…", "meta": {"topic": "配送", "issue": "届かない", ...}}

途中で止めても、同じコマンドでもう一度実行すれば続きから足す (--count は合計の件数)。
標準ライブラリだけで動く。Claude Code (`claude` コマンド) にログインしたホストで、
ワークショップの外から実行する:

    python3 scripts/phase2/generate_states.py --count 3000
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SERVICES = [
    "家電のネット通販",
    "衣料品のネット通販",
    "食品の宅配",
    "家具・インテリアの通販",
    "化粧品の定期購入",
    "動画配信のサブスクリプション",
    "携帯電話・インターネット回線",
    "銀行・決済アプリ",
    "旅行・宿泊の予約サイト",
    "チケット販売サイト",
    "フリマアプリ",
    "オンライン学習サービス",
    "スマートフォンのゲーム",
    "業務用のクラウドサービス",
    "電力・ガス",
    "保険",
]

# topic は評価セット (scripts/phase0/ja_eval.yaml) の topics と同じ
ISSUES = {
    "配送": [
        "届かない",
        "予定より遅れている",
        "別の商品や他人の荷物が届いた",
        "届け先や日時を変えたい",
        "追跡情報が更新されない",
        "不在で受け取れなかった",
        "送料や配送方法を知りたい",
    ],
    "品質": [
        "壊れた状態で届いた",
        "動かない、または不具合がある",
        "説明や写真と違う",
        "部品や付属品が足りない",
        "汚れや異物があった",
        "使い始めてすぐ壊れた",
        "サービスの品質が下がった",
    ],
    "料金": [
        "二重に請求された",
        "身に覚えのない請求がある",
        "料金プランについて知りたい",
        "返金がまだ届かない",
        "クーポンや割引が適用されなかった",
        "領収書や請求書がほしい",
        "値上げに納得できない",
        "支払い方法を変えたい",
    ],
    "操作方法": [
        "ログインできない",
        "設定の変え方がわからない",
        "機能の使い方がわからない",
        "初期設定でつまずいている",
        "機種変更でデータを移したい",
        "アプリが落ちる、または固まる",
        "画面のどこにあるかわからない",
    ],
    "その他": [
        "お礼を伝えたい",
        "要望や提案がある",
        "解約や退会をしたい",
        "登録した個人情報を変更・削除したい",
        "営業時間や店舗について知りたい",
        "担当者の対応に不満がある",
        "営業や取材の申し込み",
        "宛先を間違えた、または用件がはっきりしない",
    ],
}

EMOTIONS = [
    "怒っている",
    "いら立っている",
    "困っている",
    "落ち着いていて事務的",
    "不安そう",
    "感謝している",
    "あきれていて皮肉っぽい",
]

WANTS = [
    "返金",
    "交換",
    "修理",
    "説明や回答",
    "手続きをしてもらうこと",
    "謝罪",
    "できるだけ早い対応",
    "特に何も求めていない (報告や感想だけ)",
]

# 2〜6 文を多めにする
LENGTHS = ["1 文 (20〜40 字)", "8 文以上 (経緯を詳しく)"] + ["2〜3 文", "4〜6 文"] * 2

STYLES = [
    "丁寧なメール",
    "くだけたチャット",
    "用件だけの短い書き込み",
    "誤字や変換ミスが混じる",
    "注文番号や日付など具体的な情報が入る",
    "件名と本文に分かれている",
]

# 判断が難しい例を混ぜる (約 3 割)
TWISTS = [
    "文面は丁寧だが、強い不満が読み取れる",
    "用件が 2 つ混ざっている",
    "以前の問い合わせの続きで、返事を待たされている",
    "返金は要らないとはっきり書いている",
    "怒ってはいないと断りつつ苦情を言っている",
    "本人ではなく家族の代わりに問い合わせている",
    "質問の形だが、実際は苦情",
    "問題はすでに自分で解決していて、報告だけしている",
    "急いではいないとはっきり書いている",
]

SYSTEM = (
    "あなたは、顧客サポートに届く日本語の問い合わせ文を書く係です。"
    "機械学習の学習データに使うので、実際に人が書きそうな自然な文にしてください。"
)

PROMPT = """次の {n} 個の条件それぞれについて、顧客が書いた問い合わせ文を 1 つずつ書いてください。

- 条件の組み合わせが不自然なときは、用件を優先して自然な文にする
- 条件の言葉 (「怒っている」など) をそのまま書かず、内容と書きぶりで表す
- 書き出しや言い回しを 1 件ごとに変える。「お世話になっております」で始めるのは一部だけにする
- 実在の会社名や個人名は使わない。名前や番号が要るときは架空のものにする
- 返信や解説は書かない。顧客の文だけを書く

出力は、問い合わせ文の文字列を {n} 個並べた JSON の配列だけにしてください (条件と同じ順番)。

{specs}"""


def sample_spec(rng: random.Random) -> dict[str, str]:
    topic = rng.choice(list(ISSUES))
    spec = {
        "service": rng.choice(SERVICES),
        "topic": topic,
        "issue": rng.choice(ISSUES[topic]),
        "emotion": rng.choice(EMOTIONS),
        "wants": rng.choice(WANTS),
        "length": rng.choice(LENGTHS),
        "style": rng.choice(STYLES),
    }
    if rng.random() < 0.3:
        spec["twist"] = rng.choice(TWISTS)
    return spec


def describe(spec: dict[str, str]) -> str:
    text = (
        f"業種: {spec['service']} / 用件: {spec['topic']} ({spec['issue']}) / "
        f"感情: {spec['emotion']} / 求めること: {spec['wants']} / "
        f"長さ: {spec['length']} / 文体: {spec['style']}"
    )
    if "twist" in spec:
        text += f" / ひねり: {spec['twist']}"
    return text


def generate(specs: list[dict[str, str]], model: str, workdir: str) -> list[str]:
    """Claude Code を 1 回呼び、条件と同じ数の問い合わせ文を返す。数が合わなければ例外。"""
    lines = "\n".join(f"{i}. {describe(s)}" for i, s in enumerate(specs, 1))
    result = subprocess.run(
        ["claude", "-p", "--model", model, "--system-prompt", SYSTEM]
        + ["--tools", "", "--strict-mcp-config", "--disable-slash-commands"]
        + ["--no-session-persistence", "--output-format", "text"],
        input=PROMPT.format(n=len(specs), specs=lines),
        capture_output=True,
        text=True,
        cwd=workdir,  # プロジェクトの CLAUDE.md を読み込ませない
        check=True,
    )
    out = result.stdout
    states = json.loads(out[out.index("[") : out.rindex("]") + 1])
    if len(states) != len(specs) or not all(isinstance(s, str) and s.strip() for s in states):
        raise ValueError(f"{len(specs)} 件のはずが {len(states)} 件、または空の文があります")
    return [s.strip() for s in states]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/states.jsonl")
    p.add_argument("--count", type=int, default=3000, help="合計の件数 (すでにある行を含む)")
    p.add_argument("--batch", type=int, default=25, help="1 回の呼び出しで書かせる件数")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", default="sonnet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            seen = {json.loads(line)["state"] for line in f if line.strip()}
    missing = args.count - len(seen)
    if missing <= 0:
        print(f"すでに {len(seen)} 件あります: {out}")
        return

    # 続きから足すときに同じ条件を繰り返さないよう、乱数の種にいまの件数を混ぜる
    rng = random.Random(args.seed * 1_000_003 + len(seen))
    batches = [
        [sample_spec(rng) for _ in range(min(args.batch, missing - i))]
        for i in range(0, missing, args.batch)
    ]
    failed = 0
    with (
        tempfile.TemporaryDirectory() as workdir,
        ThreadPoolExecutor(args.workers) as pool,
        open(out, "a", encoding="utf-8") as f,
    ):
        futures = {pool.submit(generate, b, args.model, workdir): b for b in batches}
        for future in as_completed(futures):
            try:
                states = future.result()
            except (subprocess.CalledProcessError, ValueError) as e:
                failed += 1
                called = isinstance(e, subprocess.CalledProcessError)
                # 利用上限などのエラーは標準出力に出る
                detail = (e.stderr.strip() or e.stdout.strip()[-200:]) if called else e
                print(f"失敗した呼び出しを飛ばします: {detail}", file=sys.stderr)
                continue
            for state, spec in zip(states, futures[future], strict=True):
                if state not in seen:
                    seen.add(state)
                    f.write(json.dumps({"state": state, "meta": spec}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{len(seen)} / {args.count} 件", file=sys.stderr)

    print(f"{len(seen)} 件: {out}")
    if len(seen) < args.count:
        print(f"{failed} 回の呼び出しが失敗しました。もう一度実行すると続きを足します。")


if __name__ == "__main__":
    main()
