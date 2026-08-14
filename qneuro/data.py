"""PyTorch dataset and collation utilities for NeuroWorld cases."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from neuroworld import Case, NeuroWorld


class CaseDataset(Dataset[Case]):
    def __init__(self, cases: Sequence[Case]):
        self.cases = list(cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> Case:
        return self.cases[index]


def collate_cases(cases: Sequence[Case]) -> dict[str, torch.Tensor]:
    max_length = max(len(case.tokens) for case in cases)
    tokens = np.full((len(cases), max_length), NeuroWorld.pad_token, dtype=np.int64)
    mask = np.zeros((len(cases), max_length), dtype=np.bool_)
    vectors = np.stack([NeuroWorld.vector_features(case) for case in cases])
    labels = np.array([case.label for case in cases], dtype=np.int64)
    is_order = np.array([case.is_order_dependent for case in cases], dtype=np.bool_)
    order_complete = np.array([case.order_evidence_complete for case in cases], dtype=np.bool_)
    case_ids = np.array([case.case_id for case in cases], dtype=np.int64)
    for row, case in enumerate(cases):
        length = len(case.tokens)
        tokens[row, :length] = case.tokens
        mask[row, :length] = True
    return {
        "tokens": torch.from_numpy(tokens),
        "mask": torch.from_numpy(mask),
        "vector": torch.from_numpy(vectors),
        "label": torch.from_numpy(labels),
        "is_order": torch.from_numpy(is_order),
        "order_complete": torch.from_numpy(order_complete),
        "case_id": torch.from_numpy(case_ids),
    }


def shuffled_tokens(batch: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """Return a shallow batch copy with independently shuffled non-padding tokens."""

    output = dict(batch)
    tokens = batch["tokens"].clone()
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    for row in range(tokens.shape[0]):
        length = int(batch["mask"][row].sum())
        order = torch.randperm(length, generator=generator, device=tokens.device)
        tokens[row, :length] = tokens[row, :length][order]
    output["tokens"] = tokens
    return output
