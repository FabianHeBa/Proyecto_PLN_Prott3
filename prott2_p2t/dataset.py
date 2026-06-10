import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .esm_cache import cache_path


class ProteinQADataset(Dataset):
    def __init__(self, df, config: Config):
        # Muy importante resetear índices después del split
        self.data = df.reset_index(drop=True)
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # iloc busca por posición, no por nombre de columna
        row = self.data.iloc[idx]

        seq = str(row[self.config.seq_col]).strip()
        question = str(row[self.config.question_col]).strip()
        answer = str(row[self.config.answer_col]).strip()

        protein_emb = torch.load(
            cache_path(seq, self.config.cache_dir),
            map_location="cpu",
        ).float()

        return {
            "protein_emb": protein_emb,
            "question": question,
            "answer": answer,
        }


def collate_fn(batch):
    protein_embs = [x["protein_emb"] for x in batch]
    questions = [x["question"] for x in batch]
    answers = [x["answer"] for x in batch]

    max_len = max(e.shape[0] for e in protein_embs)
    dim = protein_embs[0].shape[1]

    protein_batch = torch.zeros(len(batch), max_len, dim)
    protein_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, emb in enumerate(protein_embs):
        length = emb.shape[0]
        protein_batch[i, :length] = emb
        protein_mask[i, :length] = 1

    return {
        "protein_emb": protein_batch,
        "protein_mask": protein_mask,
        "questions": questions,
        "answers": answers,
    }


def create_dataloaders(train_data_df, val_data_df, config: Config):
    train_loader = DataLoader(
        ProteinQADataset(train_data_df, config),
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        ProteinQADataset(val_data_df, config),
        batch_size=config.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
