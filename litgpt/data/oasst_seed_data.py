# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle, Alpaca
from litgpt.tokenizer import Tokenizer

@dataclass
class OASST_Seed(DataModule):
    """
    Data module for the reversed OASST‐Seed dataset.
    Each line in the provided JSONL file should be a JSON object with keys:
      - "instruction": the original human prompt,
      - "input": an empty string (or any additional context),
      - "output": the assistant response.
      
    The prompt_style is set to "alpaca" by default (which should be defined in litgpt.prompts).
    """
    # Settings for the prompt style and data splitting
    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: float = 0.1
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""

    # Local path to the reversed dataset file (JSONL format)
    json_path: Path = Path("Data/seed_data/seed_data_processed.jsonl")
    
    # Fields to be set later
    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        super().__init__()
        self.prompt_style = Alpaca()
    
    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        # For a local file, ensure it exists.
        if not self.json_path.exists():
            raise FileNotFoundError(f"Dataset file {self.json_path} not found.")
            
    def setup(self, stage: str = "") -> None:
        # Load the JSONL file (each line is a separate JSON record).
        with open(self.json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Partition the dataset into training and validation splits.
        train_data, test_data = random_split(
            data,
            [1.0 - self.val_split_fraction, self.val_split_fraction],
            generator=torch.Generator().manual_seed(self.seed),
        )
        train_data, test_data = list(train_data), list(test_data)
        
        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=False,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=False,
            ignore_index=self.ignore_index,
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

