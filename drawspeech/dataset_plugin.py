import os
import torch
import torch.nn.functional as F
import numpy as np
import json

from drawspeech.utilities.text import text_to_sequence
from drawspeech.utilities.preprocessor.preprocess_one_sample import preprocess_english
from drawspeech.utilities.tools import sketch_extractor, min_max_normalize


PITCH_MIN, ENERGY_MIN = None, None
def get_preprocessed_meta(config, dl_output, metadata):
    basename = os.path.split(metadata["wav"])[-1].replace(".wav", "")
    speaker = "LJSpeech"

    global PITCH_MIN, ENERGY_MIN
    if PITCH_MIN is None or ENERGY_MIN is None:
        # print("Loading pitch and energy stats from %s" % config["preprocessing"]["preprocessed_data"]["stats_json"])
        with open(config["preprocessing"]["preprocessed_data"]["stats_json"], "r") as f:
            stats = json.load(f)
            PITCH_MIN, pitch_max, pitch_mean, pitch_std = stats["pitch"]
            ENERGY_MIN, energy_max, energy_mean, energy_std = stats["energy"]
    
    pad_token_id = 0
    r = 2 ** (len(config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]["ch_mult"]) - 1)  # 4
    mel_pad_length = config["variables"]["latent_t_size"] * r
    phoneme_pad_length = config["preprocessing"]["phoneme_pad_length"]

    feature_level = config["preprocessing"]["preprocessed_data"]["feature"]
    if feature_level == "phoneme_level":
        pitch_pad_length = phoneme_pad_length
        energy_pad_length = phoneme_pad_length
    elif feature_level == "frame_level":
        pitch_pad_length = mel_pad_length
        energy_pad_length = mel_pad_length
    else:
        raise ValueError("Unknown feature level %s" % feature_level)
    duration_pad_length = phoneme_pad_length
    
    # load phoneme
    if "phonemes" in metadata.keys():
        phoneme_idx = torch.LongTensor(text_to_sequence(metadata["phonemes"], ["english_cleaners"]))
    else:
        assert "transcription" in metadata.keys(), "You must provide the phoneme or transcription in the metadata"
        phoneme_idx, phoneme = preprocess_english(metadata["transcription"])
        phoneme_idx = torch.LongTensor(phoneme_idx)
    phoneme_idx = F.pad(phoneme_idx, (0, phoneme_pad_length - phoneme_idx.size(0)), value=pad_token_id) if phoneme_idx.size(0) < phoneme_pad_length else phoneme_idx[:phoneme_pad_length]
    
    # load pitch and pitch sketch
    pitch_path = metadata["pitch"] if "pitch" in metadata.keys() else os.path.join(config["preprocessing"]["preprocessed_data"]["pitch"], "{}-pitch-{}.npy".format(speaker, basename))
    if os.path.exists(pitch_path):
        original_pitch = np.load(pitch_path)
        pitch = torch.from_numpy(original_pitch).float()
        pitch_length = torch.LongTensor([min(pitch.size(0), pitch_pad_length)])
        pitch = F.pad(pitch, (0, pitch_pad_length - pitch.size(0)), value=PITCH_MIN) if pitch.size(0) < pitch_pad_length else pitch[:pitch_pad_length]
    else:
        original_pitch = None
        pitch = ""
        pitch_length = ""
    
    if "pitch_sketch" in metadata.keys() and metadata["pitch_sketch"]:  # under inference mode
        assert original_pitch is None, "You cannot provide both pitch and pitch_sketch in the metadata"
        pitch_sketch_path = metadata["pitch_sketch"]
        pitch_sketch = np.load(pitch_sketch_path)
        pitch_sketch = torch.from_numpy(pitch_sketch).float()[None, None, :]
        pitch_sketch = F.interpolate(pitch_sketch, size=pitch_pad_length, mode="linear", align_corners=True).squeeze(0).squeeze(0)
        pitch_length = torch.LongTensor([min(pitch_sketch.size(0), pitch_pad_length)])
    elif original_pitch is not None:
        pitch_sketch = sketch_extractor(original_pitch)
        pitch_sketch = torch.from_numpy(pitch_sketch).float()
        pitch_sketch = F.pad(pitch_sketch, (0, pitch_pad_length - pitch_sketch.size(0)), value=PITCH_MIN) if pitch_sketch.size(0) < pitch_pad_length else pitch_sketch[:pitch_pad_length]
        pitch_sketch = min_max_normalize(pitch_sketch)
    else:
        pitch_sketch = ""

    # load energy and energy sketch
    energy_path = metadata["energy"] if "energy" in metadata.keys() else os.path.join(config["preprocessing"]["preprocessed_data"]["energy"], "{}-energy-{}.npy".format(speaker, basename))
    if os.path.exists(energy_path):
        original_energy = np.load(energy_path)
        energy = torch.from_numpy(original_energy).float()
        energy_length = torch.LongTensor([min(energy.size(0), energy_pad_length)])
        energy = F.pad(energy, (0, energy_pad_length - energy.size(0)), value=ENERGY_MIN) if energy.size(0) < energy_pad_length else energy[:energy_pad_length]
    else:
        original_energy = None
        energy = ""
        energy_length = ""
    
    if "energy_sketch" in metadata.keys() and metadata["energy_sketch"]:  # under inference mode
        assert original_energy is None, "You cannot provide both energy and energy_sketch in the metadata"
        energy_sketch_path = metadata["energy_sketch"]
        energy_sketch = np.load(energy_sketch_path)
        energy_sketch = torch.from_numpy(energy_sketch).float()[None, None, :]
        energy_sketch = F.interpolate(energy_sketch, size=energy_pad_length, mode="linear", align_corners=True).squeeze(0).squeeze(0)
        energy_length = torch.LongTensor([min(energy_sketch.size(0), energy_pad_length)])
    elif original_energy is not None:
        energy_sketch = sketch_extractor(original_energy)
        energy_sketch = torch.from_numpy(energy_sketch).float()
        energy_sketch = F.pad(energy_sketch, (0, energy_pad_length - energy_sketch.size(0)), value=ENERGY_MIN) if energy_sketch.size(0) < energy_pad_length else energy_sketch[:energy_pad_length]
        energy_sketch = min_max_normalize(energy_sketch)
    else:
        energy_sketch = ""
    
    # load phoneme duration
    duration_path = os.path.join(config["preprocessing"]["preprocessed_data"]["duration"], "{}-duration-{}.npy".format(speaker, basename))
    if os.path.exists(duration_path):
        duration = np.load(duration_path)
        duration = torch.from_numpy(duration).float()
        duration = F.pad(duration, (0, duration_pad_length - duration.size(0))) if duration.size(0) < duration_pad_length else duration[:duration_pad_length]
    else:
        duration = ""

    return {
        "phoneme_idx": phoneme_idx,
        "pitch": pitch,
        "pitch_sketch": pitch_sketch,
        "pitch_length": pitch_length,
        "energy": energy,
        "energy_sketch": energy_sketch,
        "energy_length": energy_length,
        "phoneme_duration": duration,
    }

