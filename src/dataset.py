"""
PyTorch Dataset for multilingual ASR fine-tuning

1. Read from unified manifest CSV produced by build_manifest.py (raw text)
2. Apply norm_condition text normalization once at __init__ time and cache
the result. __getitem__ returns the already-normalized transcript

Return raw audio arrays — feature extraction is implemented in collator.py.

Manifest columns: audio_path, transcript, duration_sec, dataset, split, language
==================================================================
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Union, List, Optional

from text_normalizer import get_normalizer

class ASRDataset(Dataset):
    """
    PyTorch Dataset wrapping one or more manifest CSVs.

    Args:
        manifest_paths      Path or list of paths to manifest CSV files (raw transcripts)
        norm_condition      One of "basic" / "full"
        max_duration_sec    Drop samples longer than this. Default 30s (Whisper limit)
        min_duration_sec    Drop samples shorter than this. Default 0.5s
        filter_datasets     If given, keep only rows from these dataset names
                            e.g. ["fleurs", "gigaspeech2"]
        filter_languages    If given, keep only rows from these language codes.
                            e.g. ["vi_vn"]
        max_samples         If given, truncate to the first N rows after all filters.
                            For fast smoke tests; None disables truncation.
        tokenizer           If given with max_label_tokens, drop rows whose tokenized
                            label would exceed the decoder limit. Pass for train/val
                            only — test is generation-only, so filtering it would
                            shrink the eval set and break comparability.
        max_label_tokens    Token budget for the above filter (Whisper's limit is 448).
        transform           Optional callable applied to the raw audio array in
                            __getitem__, e.g. SpecAugment. None disables it.
    """

    _VALID_NORM_CONDITIONS = ("basic", "full")

    # ==================================================================
    # __init__
    # ==================================================================
    def __init__(
        self,
        manifest_paths: Union[str, List[str]],
        norm_condition: str,
        max_duration_sec: float = 30.0,
        min_duration_sec: float = 0.5,
        filter_datasets:  Optional[List[str]] = None,
        filter_languages: Optional[List[str]] = None,
        max_samples:      Optional[int]       = None,
        tokenizer                             = None,
        max_label_tokens: Optional[int]       = None,
        transform = None,
    ):
        super().__init__()

        # Validate text normalization conditions applied
        if norm_condition not in self._VALID_NORM_CONDITIONS:
            raise ValueError(
                f"norm_condition must be one of {self._VALID_NORM_CONDITIONS}, "
                f"got {norm_condition!r}"
            )

        self.transform = transform

        # 1. Normalize manifest_paths to a list
        if isinstance(manifest_paths, str):
            manifest_paths = [manifest_paths]

        # 2. Load each CSV and concatenate
        dfs = [pd.read_csv(p) for p in manifest_paths]
        n_raw = sum(len(d) for d in dfs)
        self.df = pd.concat(dfs, ignore_index=True)

        # 3. Duration filter
        self.df = self.df[(self.df['duration_sec'] >= min_duration_sec)
                          & (self.df['duration_sec'] <= max_duration_sec)]

        # 4. Dataset filter
        if filter_datasets:
            self.df = self.df[self.df['dataset'].isin(filter_datasets)]

        # 5. Language filter
        if filter_languages:
            self.df = self.df[self.df['language'].isin(filter_languages)]

        # 6. Reset index
        self.df = self.df.reset_index(drop=True)

        # 7. Truncate to max_samples if set to keep first N rows
        if max_samples is not None and len(self.df) > max_samples:
            unique_langs = self.df["language"].unique()
            if len(unique_langs) > 1:
                n_per_lang = max(1, max_samples // len(unique_langs))
                self.df = (self.df.groupby("language", group_keys=False)
                                .head(n_per_lang)
                                .head(max_samples)
                                .reset_index(drop=True))
            else:
                self.df = self.df.head(max_samples).reset_index(drop=True)

        # 8. Apply text normalization once at construction time
        # manifest's transcript column is RAW; normalize_X is selected by
        # norm_condition.
        unique_langs = sorted(self.df["language"].unique())
        normalizers = {lang: get_normalizer(lang) for lang in unique_langs}

        texts = self.df["transcript"].fillna("").astype(str).tolist()
        langs = self.df["language"].tolist()
        self.df["transcript"] = [
            getattr(normalizers[lang], f"normalize_{norm_condition}")(t)
            for t, lang in zip(tqdm(texts, desc=f"normalize ({norm_condition})"), langs)
        ]
        self.norm_condition = norm_condition

        # 9. Label-length filter (train/val only): drops utterances whose tokenized 
        # label would exceed Whisper's 448-token decoder limit.
        if tokenizer is not None and max_label_tokens:
            N_SPECIAL = 6   # <|sot|><|lang|><|transcribe|><|notimestamps|> … <|eot|>
            enc = tokenizer(self.df["transcript"].tolist(), add_special_tokens=False).input_ids
            keep = [len(ids) + N_SPECIAL <= max_label_tokens for ids in enc]
            n_over = len(keep) - sum(keep)
            if n_over:
                print(f"[Dataset] dropped {n_over} samples exceeding {max_label_tokens} "
                      f"label tokens (Whisper decoder limit)")
                self.df = self.df[keep].reset_index(drop=True)

        # Rows removed by the duration / dataset / language filters, the
        # max_samples truncation, and the label-length filter combined.
        n_dropped = n_raw - len(self.df)
        truncated = " (truncated by max_samples)" if max_samples is not None else ""
        print(f"[Dataset] {len(self.df)} samples loaded ({norm_condition} norm), "
              f"{n_dropped} dropped by filters.{truncated}")

    # ==================================================================
    # __len__
    # ==================================================================
    def __len__(self) -> int:
        return len(self.df)

    # ==================================================================
    # __getitem__
    # ==================================================================
    def __getitem__(self, idx: int) -> dict:
        """
        Return raw audio and transcript only; no feature extraction
        """
        row = self.df.iloc[idx]

        # 1. Load audio (already 16kHz mono from preprocessing)
        audio, sr = sf.read(row['audio_path'])  # dtype=float64 by default
        if sr != 16000:
            raise ValueError(f"Expected 16kHz, got {sr}Hz for {row['audio_path']}")

        # 2. Cast to float32
        audio = audio.astype(np.float32)        # Whisper requires float32

        if self.transform is not None:          # add optional data augmentation
            audio = self.transform(audio)

        # 3. Return dict with audio and metadata
        return {
            "audio": audio,
            "transcript": row['transcript'],
            "duration": row['duration_sec'],
            "dataset": row['dataset'],
            "language": row['language'],
            "uid": os.path.splitext(os.path.basename(row['audio_path']))[0],
        }