'''
Audio Processing

1. Resample to 16kHz
2. Convert to mono channel
3. Silence trim (VAD) to remove leading/trailing silence

Input: Raw audio of any sampling rate
Output: Clean 16kHz mono numpy array
==============================================
'''

import torch
import torchaudio.transforms as T
import numpy as np
import webrtcvad

TARGET_SR = 16000
VAD_EDGE_PADDING_MS = 300   # symmetric buffer around VAD-detected speech edges

# Per-source VAD trim decision, keyed on the measured median silence ratio of each corpus (<25% -> no trim)

DATASET_CONFIG = {
    "fleurs":       {"do_trim": False},
    "common_voice": {"do_trim": True},
    "gigaspeech2":  {"do_trim": False},
}

# ==================================================================
# STEP 1: RESAMPLE
# ==================================================================
def resample(audio_array, orig_sr):
    if orig_sr == TARGET_SR:
        return np.asarray(audio_array)

    # Create resampler with T.Resample
    resampler = T.Resample(orig_freq=orig_sr, new_freq=TARGET_SR)

    # Convert audio_array to float
    audio_tensor = torch.from_numpy(np.asarray(audio_array)).float()

    # Apply resampler
    resampled = resampler(audio_tensor).numpy()
    return resampled

# ==================================================================
# STEP 2: CONVERT TO MONO CHANNEL
# ==================================================================
def to_mono(audio_array):
    # Whisper requires 1D audio input
    # Averaging both channels if inputs are stereo
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=0) # assumes torchaudio shape: (channels, samples)
    return audio_array

# ==================================================================
# STEP 3: SILENCE TRIM
# ==================================================================
def trim_silence(audio_array,
                 sr=TARGET_SR,
                 aggressiveness=2,
                 edge_padding_ms=VAD_EDGE_PADDING_MS):
    '''
    Trim leading and trailing silence using webrtcvad, keeping a buffer of
    `edge_padding_ms` milliseconds outside the first and last detected speech
    frames.
    '''
    # Step 1: VAD requires 16-bit PCM in 10/20/30ms frames
    vad = webrtcvad.Vad(aggressiveness)  # aggressiveness=2 is balanced
    frame_ms = 20
    frame_size = int(sr * frame_ms / 1000)  # 16000 × 20/1000 = 320 samples/frame

    # Step 2: Convert float32 to int16 PCM for webrtcvad
    pcm = (audio_array * 32768).astype(np.int16)

    # Step 3: Speech detection per 20ms frame
    frames = [pcm[i:i+frame_size]
              for i in range(0, len(pcm)-frame_size, frame_size)]
    is_speech = []
    for frame in frames:
        if len(frame) < frame_size:
            continue
        raw = frame.tobytes()
        is_speech.append(vad.is_speech(raw, sr))

    # Edge cases: for empty clip or all silence, return as-is
    if not frames or not any(is_speech):
        return audio_array

    # Step 4: Find speech region, then expand by edge_padding_ms on each side
    first_speech = next(i for i, v in enumerate(is_speech) if v)
    last_speech = len(is_speech) - next(i for i, v in enumerate(reversed(is_speech)) if v)

    padding_samples = int(sr * edge_padding_ms / 1000)
    start = max(0, first_speech * frame_size - padding_samples)
    end = min(len(audio_array), last_speech * frame_size + padding_samples)

    return audio_array[start:end]

# ==================================================================
# STEP 4: FULL AUDIO PROCESSING
# ==================================================================
def process_audio(audio_array,
                  sampling_rate,
                  dataset=None,
                  do_trim=True
                  ):

    if dataset and dataset in DATASET_CONFIG:
        do_trim = DATASET_CONFIG[dataset]["do_trim"]

    # Step 1: to_mono as later steps require 1D inputs
    audio_array = to_mono(audio_array)

    # Step 2: resample
    audio_array = resample(audio_array, sampling_rate)

    # Step 3: trim_silence (if flag)
    if do_trim:
        audio_array = trim_silence(audio_array)

    return audio_array
