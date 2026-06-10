import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class ProteinQFormer(nn.Module):
    def __init__(self, qformer_id, protein_dim, llm_hidden_size, num_query_tokens=32):
        super().__init__()

        q_config = AutoConfig.from_pretrained(qformer_id)
        q_config.is_decoder = True
        q_config.add_cross_attention = True
        q_config.cross_attention_hidden_size = q_config.hidden_size

        self.qformer = AutoModel.from_pretrained(
            qformer_id,
            config=q_config,
            ignore_mismatched_sizes=True,
        )

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, q_config.hidden_size) * 0.02
        )

        if protein_dim != q_config.hidden_size:
            self.protein_proj = nn.Linear(protein_dim, q_config.hidden_size)
        else:
            self.protein_proj = nn.Identity()

        self.proj_to_llm = nn.Linear(q_config.hidden_size, llm_hidden_size)

    def forward(self, protein_emb, protein_mask):
        batch_size = protein_emb.shape[0]

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_mask = torch.ones(
            batch_size,
            query_tokens.shape[1],
            dtype=torch.long,
            device=protein_emb.device,
        )

        projected_protein_emb = self.protein_proj(protein_emb)

        out = self.qformer(
            inputs_embeds=query_tokens,
            attention_mask=query_mask,
            encoder_hidden_states=projected_protein_emb,
            encoder_attention_mask=protein_mask,
        )

        protein_tokens = self.proj_to_llm(out.last_hidden_state)
        return protein_tokens
