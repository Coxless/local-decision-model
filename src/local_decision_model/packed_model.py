"""パック方式のモデル: 1 回の推論で、複数の質問の判断ロジットを返す。

NLI 学習済みの DeBERTa-v2 (mDeBERTa) の部品をそのまま使い、`embeddings` と `encoder` を
別々に呼ぶ。マスクと相対位置はグラフ内で作るので、入力は整数テンソルだけで済む
(入力の作り方は packing.py)。
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForSequenceClassification
from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position

from .scoring import decision_logits

ROLES = 3  # 状態 / 質問 / 選択肢
ROLE_FILE = "role_embeddings.safetensors"


class PackedModel(nn.Module):
    def __init__(self, model, state_sees_questions: bool = False):
        """model は DebertaV2ForSequenceClassification。

        state_sees_questions を真にすると、状態のトークンも質問を見る (比較実験用。
        質問同士は独立でなくなる)。
        """
        super().__init__()
        config = model.config
        if (
            config.model_type != "deberta-v2"
            or not getattr(config, "relative_attention", False)
            or getattr(config, "position_biased_input", True)
            or getattr(config, "conv_kernel_size", 0) > 0
        ):
            # 絶対位置の埋め込みや畳み込みがあると、位置の振り直しとマスクだけでは
            # 質問同士が独立にならない
            raise ValueError("パック方式は、相対位置のみで畳み込みのない DeBERTa-v2 が必要です")
        self.base = model
        self.state_sees_questions = state_sees_questions
        self.id2label = {int(k): v for k, v in config.id2label.items()}
        self.num_heads = config.num_attention_heads
        # 役割埋め込みは 0 で始める (学習前は元の NLI モデルと同じ計算になる)
        embeddings = model.deberta.embeddings.word_embeddings
        self.role_embeddings = nn.Embedding(ROLES, embeddings.embedding_dim)
        nn.init.zeros_(self.role_embeddings.weight)
        self.role_embeddings.to(embeddings.weight.device, embeddings.weight.dtype)

    @classmethod
    def from_pretrained(cls, model: str, state_sees_questions: bool = False) -> PackedModel:
        """Hugging Face のモデル ID か、save_pretrained で保存したディレクトリを読む。"""
        # 重みが fp16 で配布されていても fp32 で読む (torch_backend.TorchScorer と同じ)
        base = AutoModelForSequenceClassification.from_pretrained(model, dtype=torch.float32)
        packed = cls(base, state_sees_questions)
        roles = Path(model) / ROLE_FILE
        if roles.exists():
            packed.role_embeddings.load_state_dict(load_file(roles))
        return packed

    def save_pretrained(self, out_dir: str | Path) -> None:
        self.base.save_pretrained(out_dir)
        save_file(self.role_embeddings.state_dict(), Path(out_dir) / ROLE_FILE)

    def forward(self, input_ids, question_ids, option_ids, positions, marker_index):
        """入力は [B, L] の整数 (marker_index は [B, M])。判断ロジット [B, M] を返す。"""
        batch, length = input_ids.shape
        emb, encoder = self.base.deberta.embeddings, self.base.deberta.encoder

        role = (question_ids > 0).long() + (option_ids > 0).long()
        hidden = emb(
            inputs_embeds=emb.word_embeddings(input_ids) + self.role_embeddings(role),
            mask=question_ids >= 0,
        )

        # mask[b, i, j]: トークン i が j を見てよいか。
        # 状態は全員から見える。それ以外は同じ質問の中で、指示か、自分と同じ選択肢だけが見える。
        # PAD (question_ids = -1) は PAD からしか見えない
        q_i, q_j = question_ids[:, :, None], question_ids[:, None, :]
        o_i, o_j = option_ids[:, :, None], option_ids[:, None, :]
        mask = (q_j == 0) | ((q_j == q_i) & ((o_j == 0) | (o_j == o_i)))
        if self.state_sees_questions:
            mask = mask | ((q_i == 0) & (q_j > 0))

        # エンコーダは渡された相対位置をそのまま使うので、対数バケットへの変換もここで行う
        relative_pos = positions[:, :, None] - positions[:, None, :]
        if encoder.position_buckets > 0:
            relative_pos = make_log_bucket_position(
                relative_pos, encoder.position_buckets, encoder.max_relative_positions
            )
        # アテンションは 4 次元の相対位置を squeeze(0) して [B * heads, L, L] として使うので、
        # サンプルごとに違う相対位置を渡すには [1, B * heads, L, L] の形にする
        relative_pos = (
            relative_pos.long()[:, None]
            .expand(batch, self.num_heads, length, length)
            .reshape(1, batch * self.num_heads, length, length)
        )

        hidden = encoder(
            hidden, mask, output_hidden_states=False, relative_pos=relative_pos
        ).last_hidden_state

        index = marker_index[:, :, None].expand(-1, -1, hidden.size(-1))
        marked = hidden.gather(1, index).reshape(-1, 1, hidden.size(-1))
        # pooler は系列の先頭を読むので、マーカー 1 個ずつを長さ 1 の系列として渡す
        logits = self.base.classifier(self.base.dropout(self.base.pooler(marked)))
        return decision_logits(logits, self.id2label).reshape(batch, -1)
