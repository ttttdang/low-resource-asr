"""
Evaluation loop that runs generation on a DataLoader, decodes predictions,
applies text normalization, and computes WER + CER.

Functions:
  evaluate(...)         - full eval loop, returns (metrics, results_df)
  compute_metrics(...)  - jiwer WER + CER from hypothesis/reference lists
  export_trn(...)       - write reference and hypothesis files for MAPSSWE testing
==================================================================
"""
import statistics
import torch
import pandas as pd
import jiwer
from tqdm import tqdm
from typing import Literal
from text_normalizer import BaseTextNormalizer

# ==================================================================
# HELPER FUNCTION
# ==================================================================

def export_trn(results_df: pd.DataFrame, ref_path: str, hyp_path: str) -> None:
    """
    Write reference and hypothesis files in SCTK trn format for MAPSSWE testing.

    Format per line: transcript text(utterance_id)
    Utterance IDs must be unique and match across ref and hyp files.
    """
    with open(ref_path, "w", encoding="utf-8") as rf, \
         open(hyp_path, "w", encoding="utf-8") as hf:
        for _, row in results_df.iterrows():
            uid = row["uid"]
            rf.write(f"{row['reference']}({uid})\n")
            hf.write(f"{row['hypothesis']}({uid})\n")

# ==================================================================
# METRIC COMPUTATION
# ==================================================================
def compute_metrics(references, hypotheses):
    wo = jiwer.process_words(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    n_ref = wo.substitutions + wo.deletions + wo.hits   # total ref words
    return {
        "wer": wo.wer,
        "cer": cer,
        "subs": wo.substitutions,
        "ins":  wo.insertions,
        "del":  wo.deletions,
        "subs_rate": wo.substitutions / n_ref if n_ref else 0.0,
        "ins_rate":  wo.insertions    / n_ref if n_ref else 0.0,
        "del_rate":  wo.deletions     / n_ref if n_ref else 0.0,
        "n_ref_words": n_ref,
    }

# ==================================================================
# EVALUATION LOOP
# ==================================================================
def evaluate(
    model,
    dataloader,
    tokenizer,
    normalizer: dict[str, BaseTextNormalizer],
    norm_condition: Literal["basic", "full"],
    device: torch.device,
    num_beams: int = 1,
    max_new_tokens: int = 440,
    no_repeat_ngram_size: int = 0,    # 0 = OFF (Whisper/HF default)
    desc: str = "Evaluating",         # tqdm label
) -> tuple[dict, pd.DataFrame]:
    """
    Run generation over dataloader, decode predictions, normalize,
    and compute WER + CER.

    Returns:
        metrics     {wer, cer, subs, ins, del, subs_rate, ins_rate, del_rate,
                     n_ref_words, avg_gen_len, pct_at_cap}
        results_df  DataFrame with columns
                    [uid, hypothesis, reference, dataset, sample_wer, language]
    """

    # 0. Validate norm_condition and confirm it matches the dataset's
    valid_norms = ("basic", "full")
    if norm_condition not in valid_norms:
        raise ValueError(
            f"norm_condition must be one of {valid_norms}, got {norm_condition!r}"
        )

    ds_norm = getattr(dataloader.dataset, "norm_condition", None)
    if ds_norm is not None and ds_norm != norm_condition:
        raise ValueError(
            f"norm_condition mismatch: dataset was constructed with {ds_norm!r}, "
            f"but evaluate() was called with {norm_condition!r}. References were "
            f"pre-normalized using the dataset value; hypotheses would be normalized "
            f"using evaluate()'s value — WER would compare mismatched pipelines."
        )

    # 1. Set model to eval mode
    model.eval()

    # Accumulators
    all_hypotheses = []
    all_references = []
    all_datasets = []
    all_uids = []
    all_gen_lens = []          # token-length per sample (for hallucination detection)
    all_langs = []

    # 2. Disable gradient computation for the entire loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):

            # 3. Move input_features to device
            input_features = batch["input_features"].to(device)

            # 4. Generate predicted token IDs
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                predicted_ids = model.generate(
                    input_features,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )

            # Diagnostic: track generated length
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            gen_lens = (predicted_ids != pad_id).sum(dim=-1).cpu().tolist()
            all_gen_lens.extend(gen_lens)

            # 5. Decode predictions to strings
            hypotheses = tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            # 6. References are pre-normalized using the dataset's norm_condition
            references = batch["transcript"]

            # 7. Normalize hypotheses per-row using the row's language
            langs = batch["language"]
            hypotheses = [
                getattr(normalizer[lang], f"normalize_{norm_condition}")(h)
                for h, lang in zip(hypotheses, langs)
            ]

            # 8. Accumulate
            all_hypotheses.extend(hypotheses)
            all_references.extend(references)
            all_datasets.extend(batch["dataset"])
            all_langs.extend(batch["language"])
            all_uids.extend(batch["uid"])

    # Diagnostic: log first 2 (hyp, ref) pairs for visual inspection of failure mode
    # hallucination = repeated tokens or random vocab
    for i in range(min(2, len(all_hypotheses))):
        print(f"[sample {i}] pred='{all_hypotheses[i][:120]}'")
        print(f"[sample {i}]  ref='{all_references[i][:120]}'")

    # Post-loop diagnostic: flag empty references and hypotheses from aggressive normalization
    n_empty_refs = sum(1 for r in all_references if not r.strip())
    n_empty_hyps = sum(1 for h in all_hypotheses if not h.strip())
    if n_empty_refs or n_empty_hyps:
        print(f"[warn] empty after normalization: {n_empty_refs} refs, {n_empty_hyps} hyps")

    # 9. Compute aggregate metrics
    metrics = compute_metrics(all_references, all_hypotheses)

    metrics["avg_gen_len"] = statistics.mean(all_gen_lens) if all_gen_lens else 0.0
    metrics["pct_at_cap"]  = sum(1 for L in all_gen_lens if L >= max_new_tokens - 5) / max(len(all_gen_lens), 1)

    # 10. Build results DataFrame
    results_df = pd.DataFrame({
        "uid"        : all_uids,
        "hypothesis" : all_hypotheses,
        "reference"  : all_references,
        "dataset"    : all_datasets,
        "sample_wer" : [jiwer.wer([r], [h]) if r.strip() else float("nan")
                        for h, r in zip(all_hypotheses, all_references)],
        "language"   : all_langs,
    })

    # 11. Per-dataset WER breakdown (print only, for logging)
    print(f"\n{'='*50}")
    print(f"Overall  WER: {metrics['wer']:.4f}  CER: {metrics['cer']:.4f}  "
      f"SUB: {metrics['subs_rate']:.4f}  INS: {metrics['ins_rate']:.4f}  "
      f"DEL: {metrics['del_rate']:.4f}  (N={metrics['n_ref_words']})")
    print(f"{'='*50}")
    for ds, grp in results_df.groupby("dataset"):
        ds_metrics = compute_metrics(
            grp["reference"].tolist(),
            grp["hypothesis"].tolist(),
            )
        print(f"  {ds:20s}  WER: {ds_metrics['wer']:.4f}  CER: {ds_metrics['cer']:.4f}  "
        f"SUB: {ds_metrics['subs_rate']:.4f}  INS: {ds_metrics['ins_rate']:.4f}  "
        f"DEL: {ds_metrics['del_rate']:.4f}  (N={ds_metrics['n_ref_words']})")

    return metrics, results_df