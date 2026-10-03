"""パック入力: 状態と複数の質問を 1 本の系列に詰める (numpy だけで書く)。

    [CLS] 状態 [SEP] | [Q] 質問1 [SEP] | choice の指示 [O] 選択肢A [SEP] [O] 選択肢B [SEP] | PAD

- マーカー [Q] / [O] は [CLS] と同じトークンで、その位置の隠れ状態から判断ロジットを読む
- 見える範囲: 状態 → 状態のみ / 質問 → 状態 + 自分の質問 / 選択肢 → 状態 + 親の質問 + 自分。
  モデル側 (packed_model.py) が ``question_ids`` と ``option_ids`` からマスクを作る
- 位置番号は質問ごとに「状態のすぐ後ろ」から振り直す。選択肢は親の質問の後ろから振る
- 状態の切り詰め長は ``max_state_tokens`` で固定する。他の質問の有無で状態の長さが変わると、
  質問同士が独立でなくなるため
- 1 つのパックに入りきらない質問や選択肢は、次のパックに回す
  (状態と choice の指示はパックごとに繰り返す)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Choice, Noul, Question
from .scoring import S1Config

MarkerKey = tuple[str, int]  # (質問名, 選択肢の番号。noul は 0)


@dataclass
class Pack:
    """1 回の推論の入力。配列の長さは input_ids などが L、marker_index が max_markers。"""

    input_ids: np.ndarray
    question_ids: np.ndarray  # 0 = 状態、1 以上 = 質問、-1 = PAD
    option_ids: np.ndarray  # 0 = 選択肢ではない、1 以上 = 選択肢
    positions: np.ndarray
    marker_index: np.ndarray  # 使わない枠は 0
    markers: list[MarkerKey]  # マーカーの枠 → (質問名, 選択肢の番号)

    def arrays(self) -> dict[str, np.ndarray]:
        """モデルに渡す配列 (バッチの次元つき)。"""
        names = ("input_ids", "question_ids", "option_ids", "positions", "marker_index")
        return {n: getattr(self, n)[None] for n in names}


@dataclass
class _Group:
    """互いに見える範囲がひとまとまりの単位。noul は head だけ、choice は head + 選択肢。"""

    head: list[int]
    head_key: MarkerKey | None  # None でなければ head にマーカーを付けて読む
    options: list[tuple[list[int], MarkerKey]]

    @property
    def head_cost(self) -> int:
        return len(self.head) + (2 if self.head_key else 0)


def _groups(encode, questions: dict[str, Question], choice_layout: str) -> list[_Group]:
    groups = []
    for name, q in questions.items():
        if isinstance(q, Noul):
            groups.append(_Group(encode(q.instructions), (name, 0), []))
        elif isinstance(q, Choice) and choice_layout == "expanded":
            # 選択肢ごとの仮説文を、独立した質問として詰める (ペア方式と同じ文になる)
            for i, hyp in enumerate(q.hypotheses()):
                groups.append(_Group(encode(hyp), (name, i), []))
        elif isinstance(q, Choice):
            # {option} より前を共有し、選択肢の側に {option} 以降をつなげる
            prefix, _, suffix = q.instructions.partition("{option}")
            options = [
                (encode(o + suffix.replace("{option}", o)), (name, i))
                for i, o in enumerate(q.options)
            ]
            groups.append(_Group(encode(prefix), None, options))
    return groups


class _Builder:
    """1 回分の系列を組み立てる。"""

    def __init__(self, state_ids: list[int], config: S1Config, cls_id: int, sep_id: int):
        self.config = config
        self.cls_id, self.sep_id = cls_id, sep_id
        self.state_len = len(state_ids)
        self.input_ids = list(state_ids)
        self.question_ids = [0] * len(state_ids)
        self.option_ids = [0] * len(state_ids)
        self.positions = list(range(len(state_ids)))
        self.marker_index: list[int] = []
        self.markers: list[MarkerKey] = []
        self.n_questions = 0
        self.n_options = 0

    def fits(self, tokens: int, markers: int) -> bool:
        return (
            len(self.input_ids) + tokens <= max(self.config.lengths)
            and len(self.markers) + markers <= self.config.max_markers
        )

    def _extend(self, ids: list[int], option_id: int, start: int) -> None:
        self.input_ids += ids
        self.question_ids += [self.n_questions] * len(ids)
        self.option_ids += [option_id] * len(ids)
        self.positions += range(start, start + len(ids))

    def _add_read(self, ids: list[int], key: MarkerKey, option_id: int, start: int) -> None:
        """マーカー + ids + [SEP] を足す。ids の位置番号は start から。"""
        self.marker_index.append(len(self.input_ids))
        self.markers.append(key)
        if self.config.marker_position == "cls":
            # ペア方式の [CLS] と同じ位置 (0) に置く
            self._extend([self.cls_id], option_id, 0)
        else:
            self._extend([self.cls_id], option_id, start)
            start += 1
        self._extend(ids + [self.sep_id], option_id, start)

    def add_head(self, group: _Group) -> None:
        self.n_questions += 1
        if group.head_key:
            self._add_read(group.head, group.head_key, 0, self.state_len)
        else:
            self._extend(group.head, 0, self.state_len)

    def add_option(self, group: _Group, ids: list[int], key: MarkerKey) -> None:
        self.n_options += 1
        self._add_read(ids, key, self.n_options, self.state_len + len(group.head))

    def build(self, pad_id: int) -> Pack:
        used = len(self.input_ids)
        length = min(n for n in self.config.lengths if n >= used)
        pad = length - used

        def arr(values: list[int], fill: int) -> np.ndarray:
            return np.array(values + [fill] * pad, dtype=np.int64)

        marker_index = np.zeros(self.config.max_markers, dtype=np.int64)
        marker_index[: len(self.marker_index)] = self.marker_index
        return Pack(
            arr(self.input_ids, pad_id),
            arr(self.question_ids, -1),
            arr(self.option_ids, 0),
            arr(self.positions, 0),
            marker_index,
            self.markers,
        )


def pack(tokenizer, state: str, questions: dict[str, Question], config: S1Config) -> list[Pack]:
    """状態と質問を、1 回以上の推論の入力にする。"""
    if config.max_state_tokens >= max(config.lengths):
        raise ValueError("max_state_tokens は lengths の最大値より小さくしてください")

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    state_ids = [cls_id] + encode(state)[: config.max_state_tokens - 2] + [sep_id]

    def new_builder() -> _Builder:
        return _Builder(state_ids, config, cls_id, sep_id)

    packs: list[Pack] = []
    builder = new_builder()
    for group in _groups(encode, questions, config.choice_layout):
        options = list(group.options)
        while True:
            # head と、選択肢があれば少なくとも 1 つが入ること
            tokens = group.head_cost + (len(options[0][0]) + 2 if options else 0)
            if not builder.fits(tokens, 1):
                if not builder.markers:
                    name = (group.head_key or options[0][1])[0]
                    raise ValueError(f"質問 {name!r} が長すぎて 1 回の推論に入りません")
                packs.append(builder.build(tokenizer.pad_token_id))
                builder = new_builder()
                continue
            builder.add_head(group)
            while options and builder.fits(len(options[0][0]) + 2, 1):
                builder.add_option(group, *options.pop(0))
            if not options:
                break
            # 残りの選択肢は次のパックへ (指示は繰り返す)
            packs.append(builder.build(tokenizer.pad_token_id))
            builder = new_builder()
    if builder.markers:
        packs.append(builder.build(tokenizer.pad_token_id))
    return packs


def unpack(
    packs: list[Pack], outputs: list[np.ndarray], questions: dict[str, Question]
) -> dict[str, np.ndarray]:
    """各パックのマーカーごとの判断ロジットを、質問名 → ロジット (choice は選択肢の順) に戻す。"""
    z = {
        name: np.full(len(q.options) if isinstance(q, Choice) else 1, np.nan)
        for name, q in questions.items()
    }
    for p, out in zip(packs, outputs, strict=True):
        for marker_slot, (name, index) in enumerate(p.markers):
            z[name][index] = out[marker_slot]
    return z
