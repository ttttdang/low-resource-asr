"""
Data collator for Whisper fine-tuning.

Converts a batch of raw audio arrays + transcripts into:
  - input_features : (batch, 80, 3000)  log-mel spectrogram
  - labels         : (batch, seq_len)   token IDs, padding replaced with -100
==================================================================
"""

from dataclasses import dataclass
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from torch.nn.utils.rnn import pad_sequence

# Corpus language code -> Whisper language name. A language Whisper has no token
# for borrows the closest available one, as a proxy
_LANG_TO_WHISPER = {
    "vi_vn":       "vietnamese",
    "cmn_hans_cn": "chinese",
    "fr_fr":       "french",
    "ast_es":      "spanish",   # Whisper has no Asturian token; <|es|> proxy
    "es_419":      "spanish",
}

@dataclass
class WhisperCollator:
    """
    Callable collator for use as DataLoader(collate_fn=...)
    
    Args:
        feature_extractor : WhisperFeatureExtractor instance
        tokenizer         : WhisperTokenizer instance
        language          : Forced decoding language e.g. "vietnamese"
        task              : "transcribe"
     
    """
    feature_extractor : WhisperFeatureExtractor
    tokenizer         : WhisperTokenizer
    language          : str = "vietnamese"
    task              : str = "transcribe"

    def __post_init__(self):
        # Set forced prefix tokens (language + task) once at construction.
        self.tokenizer.set_prefix_tokens(language=self.language, task=self.task)
        self._sot_id = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")


    def __call__(self, batch: list[dict]) -> dict:

        # ==================================================================
        # STEP 1. AUDIO
        # ==================================================================

        # 1. Extract raw audio arrays from each sample in the batch
        audio_arrays = [b['audio'] for b in batch]

        # 2. Run feature extractor to obtain input_features
        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )
        input_features = inputs.input_features

        # ==================================================================
        # STEP 2. LABELS
        # ==================================================================

        # 3. Extract transcripts and metadata from batch
        transcripts = [b['transcript'] for b in batch]
        datasets    = [b['dataset']    for b in batch]
        uids        = [b['uid']        for b in batch]
        languages   = [b['language']   for b in batch]

        # 4. Per-sample tokenization — each sample gets its own language prefix token
        all_ids, all_masks = [], []
        for transcript, lang in zip(transcripts, languages):
            whisper_lang = _LANG_TO_WHISPER.get(lang)
            if whisper_lang is None:
                raise ValueError(
                    f"No Whisper language mapping for language={lang!r}. Add an entry "
                    f"to _LANG_TO_WHISPER in src/collator.py mapping it to a Whisper "
                    f"language name (or a proxy, e.g. Asturian -> 'spanish'). "
                    f"Known: {sorted(_LANG_TO_WHISPER)}"
                )
            self.tokenizer.set_prefix_tokens(language=whisper_lang, task=self.task)
            enc = self.tokenizer(transcript, return_tensors="pt", padding=False)
            ids, mask = enc.input_ids[0], enc.attention_mask[0]
            # Drop leading <|startoftranscript|>: the model re-adds it via
            # shift_tokens_right, so keeping it doubles SOT and misaligns labels.
            if ids[0].item() == self._sot_id:
                ids, mask = ids[1:], mask[1:]

            if ids[0].item() == self._sot_id:
                ids, mask = ids[1:], mask[1:]
            all_ids.append(ids)
            all_masks.append(mask)

        # 5. Pad to longest in batch, mask padding with -100
        pad_id = self.tokenizer.pad_token_id
        padded   = pad_sequence(all_ids,   batch_first=True, padding_value=pad_id)
        attn_pad = pad_sequence(all_masks, batch_first=True, padding_value=0)
        labels   = padded.masked_fill(attn_pad.ne(1), -100)
        
        # ==================================================================
        # STEP 3. RETURN
        # ==================================================================
        return {
            "input_features": input_features,   # (batch, 80, 3000)
            "labels":         labels,            # (batch, max_label_len)
            "transcript":     transcripts,       # List[str], length=batch
            "dataset":        datasets,          # List[str], length=batch
            "uid":            uids,              # List[str], length=batch
            "language":       languages,         # passes through to evaluate()
        }
