"""
Build Manifest

Build unified CSV for each corpus and store raw transcripts, after audio is pre-processed.
Audio processing is skipped if the processed WAV already exists.

Output: {manifest_dir}/{language}/{split}.csv

Usage:
  python preprocessing/build_manifest.py --language vi_vn --split all
  python preprocessing/build_manifest.py --language cmn_hans_cn --split train --num-proc 4

  # Merge per-language corpora into one combined CSV
  python preprocessing/build_manifest.py --merge --languages vi_vn cmn_hans_cn fr_fr
"""

import os, argparse
import pandas as pd
import soundfile as sf
from datasets import load_from_disk, Dataset, Audio

import audio_processor
from audio_processor import process_audio

# ==================================================================
# DATA PATHS
# ==================================================================
# Defaults are relative to the repo root. Override any of them with
# --data-dir / --audio-dir / --manifest-dir. All three are resolved to
# absolute paths in __main__ before workers fork, because the resolved
# AUDIO_DIR is written verbatim into the manifest's `audio_path` column and
# must stay valid regardless of the training job's working directory.
DATA_DIR     = "./data"
AUDIO_DIR    = "./processed_audio"
MANIFEST_DIR = "./manifests"


# Language code: Output WAV dirs are keyed on the dataset name
LANG_CODE = {
    "fleurs":          {"vi_vn": "vi_vn", 
                        "cmn_hans_cn": "cmn_hans_cn",
                        "fr_fr": "fr_fr", 
                        "ast_es": "ast_es",
                        "es_419": "es_419"},
    "common_voice":    {"vi_vn": "vi"},
    "common_voice_25": {"cmn_hans_cn": "zh-CN", 
                        "fr_fr": "fr", 
                        "es_419": "es"},
    "gigaspeech2":     {"vi_vn": "vi"},
}


# ==================================================================
# AUDIO HELPER
# ==================================================================
def save_processed_audio(audio_array, dataset, language, split, uid):
    """Save processed WAV (skip if exists) and return its path.
    """
    out_dir = os.path.join(AUDIO_DIR, dataset, language, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uid}.wav")
    if os.path.exists(out_path):
        return out_path
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    # Explicit format/subtype because the temp filename ends in ".tmp.<pid>", not ".wav" 
    sf.write(tmp_path, audio_array, 16000, format='WAV', subtype='PCM_16')
    os.rename(tmp_path, out_path)
    return out_path


def _wav_duration_or_none(out_path, target_sr=16000):
    """Returns WAV duration in seconds if the file exists and parses cleanly.
    """
    if not os.path.exists(out_path):
        return None
    try:
        return sf.info(out_path).frames / target_sr
    except Exception as e:
        print(f"[warn] sf.info failed on {out_path}: {e!r}; will reprocess",
              flush=True)
        return None


# ==================================================================
# PER-SAMPLE PROCESSORS
# ==================================================================
# Each processor takes (example, idx, language, split) and follow
#   1. Compute the deterministic uid + output path.
#   2. Skip if the WAV already exists, otherwise process audio
#   3. Return the row dict for the unified manifest.

def _process_fleurs_sample(example, idx, language, split):
    uid = f"fleurs_{language}_{split}_{idx:06d}"
    out_path = os.path.join(AUDIO_DIR, "fleurs", language, split, f"{uid}.wav")
    duration_sec = _wav_duration_or_none(out_path)
    if duration_sec is None:
        processed = process_audio(
            example["audio"]["array"],
            example["audio"]["sampling_rate"],
            dataset="fleurs",
        )
        save_processed_audio(processed, "fleurs", language, split, uid)
        duration_sec = len(processed) / 16000
    return {
        "audio_path":   out_path,
        "client_id":    "",  # FLEURS does not have speaker IDs
        "transcript":   example.get("transcription") or "",
        "duration_sec": duration_sec,
        "dataset":      "fleurs",
        "split":        split,
        "language":     language,
    }

def _process_common_voice_sample(example, idx, language, split):
    """Common Voice dataset for Vietnamese"""
    uid = f"common_voice_{language}_{split}_{idx:06d}"
    out_path = os.path.join(AUDIO_DIR, "common_voice", language, split, f"{uid}.wav")
    duration_sec = _wav_duration_or_none(out_path)
    if duration_sec is None:
        processed = process_audio(
            example["audio"]["array"],
            example["audio"]["sampling_rate"],
            dataset="common_voice",
        )
        save_processed_audio(processed, "common_voice", language, split, uid)
        duration_sec = len(processed) / 16000
    return {
        "audio_path":   out_path,
        "client_id":    example.get("client_id") or "",
        "transcript":   example.get("sentence") or "",
        "duration_sec": duration_sec,
        "dataset":      "common_voice",
        "split":        split,
        "language":     language,
    }


def _process_common_voice_25_sample(example, idx, language, split):
    """Common Voice datasets for Mandarin, French, and Spanish."""
    uid = f"common_voice_25_{language}_{split}_{idx:06d}"
    out_path = os.path.join(AUDIO_DIR, "common_voice_25", language, split, f"{uid}.wav")
    duration_sec = _wav_duration_or_none(out_path)
    if duration_sec is None:
        processed = process_audio(
            example["audio"]["array"],
            example["audio"]["sampling_rate"],
            dataset="common_voice",
        )
        save_processed_audio(processed, "common_voice_25", language, split, uid)
        duration_sec = len(processed) / 16000
    return {
        "audio_path":   out_path,
        "client_id":    example.get("client_id") or "",
        "transcript":   example.get("sentence") or "",
        "duration_sec": duration_sec,
        "dataset":      "common_voice_25",
        "split":        split,
        "language":     language,
    }


def _process_gigaspeech2_sample(example, idx, language, split):
    """GS2 reads audio from a path string in the CSV manifest (no HF audio
    column), so sf.read happens inline per worker."""
    uid = os.path.basename(example["audio_path"]).replace(".wav", "")
    out_path = os.path.join(AUDIO_DIR, "gigaspeech2", language, split, f"{uid}.wav")
    duration_sec = _wav_duration_or_none(out_path)
    if duration_sec is None:
        audio_array, sr = sf.read(example["audio_path"])
        processed = process_audio(audio_array, sr, dataset="gigaspeech2")
        save_processed_audio(processed, "gigaspeech2", language, split, uid)
        duration_sec = len(processed) / 16000
    return {
        "audio_path":   out_path,
        "client_id":    "",  # GS2 manifest does have carry speaker IDs
        "transcript":   example["transcript"] or "",
        "duration_sec": duration_sec,
        "dataset":      "gigaspeech2",
        "split":        split,
        "language":     language,
    }


# ==================================================================
# DATASET LOADERS — use ds.map(num_proc=N) for parallel audio processing
# ==================================================================
def _run_map(ds, fn, language, split, num_proc, desc):
    """Returns a pandas DataFrame in the unified schema.
    """
    
    result_ds = ds.map(
        fn,
        with_indices=True,
        fn_kwargs={"language": language, "split": split},
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc=desc,
    )
    return result_ds.to_pandas()


def load_fleurs(split, language, num_proc):
    lang_dir = LANG_CODE["fleurs"][language]
    path = os.path.join(DATA_DIR, f"fleurs/{lang_dir}/{split}")
    ds = load_from_disk(path)
    return _run_map(ds, _process_fleurs_sample, language, split,
                    num_proc, f"FLEURS {language}/{split}")


def load_common_voice(split, language, num_proc):
    lang_dir = LANG_CODE["common_voice"][language]
    path = os.path.join(DATA_DIR, f"common_voice/{lang_dir}/{split}")
    ds = load_from_disk(path)
    ds = ds.cast_column("audio", Audio())
    return _run_map(ds, _process_common_voice_sample, language, split, 
                    num_proc, f"CV17 {language}/{split}")


def load_common_voice_25(split, language, num_proc):
    lang_dir = LANG_CODE["common_voice_25"][language]
    if split == "train":
        path = os.path.join(DATA_DIR, f"common_voice_25/{lang_dir}/train_balanced")
    else:
        path = os.path.join(DATA_DIR, f"common_voice_25/{lang_dir}/{split}")
    ds = load_from_disk(path)
    ds = ds.cast_column("audio", Audio())
    return _run_map(ds, _process_common_voice_25_sample, language, split, 
                    num_proc, f"CV25 {language}/{split}")


def load_gigaspeech2(split, language, num_proc):
    lang_dir = LANG_CODE["gigaspeech2"][language]
    if split == "train":
        manifest_path = os.path.join(
            DATA_DIR, f"gigaspeech2/data/{lang_dir}/train_50h_manifest.csv")
    else:
        manifest_path = os.path.join(
            DATA_DIR, f"gigaspeech2/data/{lang_dir}/{split}_manifest.csv")
    df = pd.read_csv(manifest_path)
    ds = Dataset.from_pandas(df)
    return _run_map(ds, _process_gigaspeech2_sample, language, split, 
                    num_proc, f"GS2 {language}/{split}")


# ==================================================================
# PER-LANGUAGE MANIFEST
# ==================================================================
def build_language_manifest(language, splits, num_proc, source_filter):
    """Build {split}_{language}.csv by concatenating all applicable sources
    for that language.
    """
    def _ds_filter(src):
        return source_filter in ("all", src)

    for split in splits:
        print(f"\n{'='*50}")
        print(f"Building {language} {split} manifest "
              f"(num_proc={num_proc}, source={source_filter})")
        print(f"{'='*50}")

        dfs = []

        # FLEURS for all languages
        if _ds_filter("fleurs") and language in LANG_CODE["fleurs"]:
            try:
                dfs.append(load_fleurs(split, language, num_proc))
            except FileNotFoundError as e:
                print(f"Skipping FLEURS: {e}")

        # CV for Vietnamese
        if _ds_filter("common_voice") and language in LANG_CODE["common_voice"]:
            try:
                dfs.append(load_common_voice(split, language, num_proc))
            except FileNotFoundError as e:
                print(f"Skipping CV17: {e}")

        # CV2 for Mandarin, French, and Spanish
        if _ds_filter("common_voice_25") and language in LANG_CODE["common_voice_25"]:
            try:
                dfs.append(load_common_voice_25(split, language, num_proc))
            except FileNotFoundError as e:
                print(f"Skipping CV25: {e}")

        # GigaSpeech2 for Vietnamese
        if _ds_filter("gigaspeech2") and language in LANG_CODE["gigaspeech2"]:
            try:
                gs_split = "dev" if split == "validation" else split
                gs_df = load_gigaspeech2(gs_split, language, num_proc)
                gs_df["split"] = split
                dfs.append(gs_df)
            except FileNotFoundError as e:
                print(f"Skipping GigaSpeech2: {e}")

        if not dfs:
            print(f"No data loaded for {language} {split}")
            continue

        manifest = pd.concat(dfs, ignore_index=True)
        out_dir = os.path.join(MANIFEST_DIR, language)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{split}.csv")
        if source_filter != "all":
            out_csv = os.path.join(out_dir, f"{split}.{source_filter}.csv")
        manifest.to_csv(out_csv, index=False)

        total_hours = manifest["duration_sec"].sum() / 3600
        print(f"Saved {out_csv}: {len(manifest)} samples, {total_hours:.2f} h")
        print(manifest.groupby("dataset")["duration_sec"]
              .agg(samples="count", hours=lambda s: s.sum()/3600))


# ==================================================================
# MERGE MANIFEST
# ==================================================================
def merge_manifests(languages, split):
    """Concatenate per-language manifests into one combined CSV."""
    dfs = []
    for lang in languages:
        path = os.path.join(MANIFEST_DIR, lang, f"{split}.csv")
        if not os.path.exists(path):
            print(f"  [skip] missing: {path}")
            continue
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  {lang}: {len(df)} samples, {df['duration_sec'].sum()/3600:.2f} h")

    if not dfs:
        print(f"No manifests found for split={split}")
        return

    combined = pd.concat(dfs, ignore_index=True)
    out_csv = os.path.join(MANIFEST_DIR, f"{split}_combined.csv")
    combined.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}: {len(combined)} samples, "
          f"{combined['duration_sec'].sum()/3600:.2f} h total")


# ==================================================================
# __MAIN__
# ==================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="vi_vn",
                        choices=["vi_vn", "cmn_hans_cn", "fr_fr", "es_419", "ast_es"])
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "validation", "test", "all"])
    parser.add_argument("--num-proc", type=int, default=4,
                        help="Worker processes. Default 4; raise to match the number of CPU cores available to the job.")
    parser.add_argument("--source", type=str, default="all",
                        choices=["all", "fleurs", "common_voice",
                                 "common_voice_25", "gigaspeech2"],
                        help="Process only one source for testing. Default 'all' "
                             "= every applicable source for the language.")
    parser.add_argument("--no-trim", action="store_true")

    parser.add_argument("--data-dir", type=str, default=None,
                        help=f"Root directory holding the downloaded source corpora. "
                             f"Default {DATA_DIR!r} (relative to the repo root).")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help=f"Root directory for processed WAV outputs. "
                             f"Default {AUDIO_DIR!r}.")
    parser.add_argument("--manifest-dir", type=str, default=None,
                        help=f"Root directory for output manifests. Default "
                             f"{MANIFEST_DIR!r}. Per-language manifests are written "
                             f"to {{manifest_dir}}/{{lang}}/{{split}}.csv.")
    parser.add_argument("--merge", action="store_true",
                        help="Merge per-language manifests into one combined CSV")
    parser.add_argument("--languages", nargs="+",
                        help="Languages to merge e.g. vi_vn cmn_hans_cn fr_fr")
    args = parser.parse_args()

    if args.no_trim and (args.audio_dir is None or args.manifest_dir is None):
        raise SystemExit(
            "--no-trim requires both --audio-dir and --manifest-dir.")

    # Resolve paths + DATASET_CONFIG before any worker forks.
    # abspath: AUDIO_DIR is written into the manifest's `audio_path` column and
    # read directly by dataset.py, so it must not depend on the directory the
    # training job happens to be launched from.
    DATA_DIR     = os.path.abspath(os.path.expanduser(args.data_dir     or DATA_DIR))
    AUDIO_DIR    = os.path.abspath(os.path.expanduser(args.audio_dir    or AUDIO_DIR))
    MANIFEST_DIR = os.path.abspath(os.path.expanduser(args.manifest_dir or MANIFEST_DIR))

    # Ablation override. Rebind to a NEW dict rather than mutating audio_processor's
    # module-level constant in place.
    if args.no_trim:
        audio_processor.DATASET_CONFIG = {
            src: {"do_trim": profile["do_trim"] and not args.no_trim}
            for src, profile in audio_processor.DATASET_CONFIG.items()
        }

    os.makedirs(AUDIO_DIR,    exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    print(f"DATA_DIR={DATA_DIR}  AUDIO_DIR={AUDIO_DIR}  "
          f"MANIFEST_DIR={MANIFEST_DIR}  no_trim={args.no_trim}", flush=True)
    print(f"effective DATASET_CONFIG={audio_processor.DATASET_CONFIG}", flush=True)


    splits = (["train", "validation", "test"]
              if args.split == "all" else [args.split])

    if args.merge:
        for split in splits:
            merge_manifests(args.languages, split)
    else:
        build_language_manifest(args.language, splits, args.num_proc, args.source)
