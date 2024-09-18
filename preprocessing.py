import os
import json
import re
import random
import librosa
from g2p_en import G2p
from tqdm import tqdm
import numpy as np
import torch

from drawspeech.utilities.preprocessor.preprocess_one_sample import preprocess_english
from drawspeech.utilities.tools import sketch_extractor

def format_ljspeech(dataset_root="/mnt/users/hccl.local/wdchen/dataset/LJSpeech-1.1"):
    ''' Create json file for ljspeech dataset. 
        Provide the path where you save the LJSpeech dataset and let the program do the rest.
    '''
    
    metadata = os.path.join(dataset_root, "metadata.csv")
    wav_path = os.path.join(dataset_root, "wavs")

    # if not os.path.exists("data/dataset/ljspeech"):
    #     os.makedirs("data/dataset/ljspeech")
    # if not os.path.exists("data/dataset/ljspeech/wavs"):
    #     cmd = f"ln -s {wav_path} data/dataset/ljspeech/wavs"
    #     os.system(cmd)

    data = []
    print("Start formating ljspeech dataset.")
    with open(os.path.join(metadata), encoding="utf-8") as f:
        for line in f:
            id, text, norm_text = line.strip().split("|")
            norm_text = re.sub(r'\(|\)|\[|\]|<.*?>', '', norm_text)  # remove (), [], <xxx>

            file_path = f"wavs/{id}.wav"
            duration = librosa.get_duration(filename=os.path.join(dataset_root, file_path))
             
            data.append(
                {
                    "wav": file_path,
                    "transcription": norm_text,
                    "duration": duration
                }
            )
    
    # print("Perform text to phoneme conversion.")
    # g2p = G2p()
    # for d in tqdm(data, total=len(data)):
    #     phoneme_idx, phoneme = preprocess_english(d["transcription"], always_use_g2p=True, g2p=g2p, verbose=False)
    #     d["phonemes"] = phoneme

    num_data = len(data)
    print(f"Total {num_data} data in ljspeech dataset.")
    ids = list(range(num_data))
    random.shuffle(ids)

    train_ids = ids[:12500]
    val_ids = ids[12500:12800]
    test_ids = ids[12800:]

    if not os.path.exists("data/dataset/metadata/ljspeech"):
        os.makedirs("data/dataset/metadata/ljspeech")
    
    json.dump({"data": [data[i] for i in train_ids]}, open(os.path.join("data/dataset/metadata/ljspeech", "train.json"), "w"), indent=1, ensure_ascii=False)
    json.dump({"data": [data[i] for i in val_ids]}, open(os.path.join("data/dataset/metadata/ljspeech", "val.json"), "w"), indent=1, ensure_ascii=False)
    json.dump({"data": [data[i] for i in test_ids]}, open(os.path.join("data/dataset/metadata/ljspeech", "test.json"), "w"), indent=1, ensure_ascii=False)
    print("Finish formating ljspeech dataset.")


def add_phoneme_for_ljspeech(ljspeech_json_path = "data/dataset/metadata/ljspeech", fs_files = ["data/dataset/metadata/ljspeech/phoneme_level/metadata.txt"]):
    print("Add phoneme for ljspeech dataset.")
    names, phones = [], []
    for fs_file in fs_files:
        for line in open(fs_file, "r").readlines():
            fname, speaker, p, raw_text = line.strip().split("|")
            names.append(fname)
            phones.append(p)        

    json_files = os.listdir(ljspeech_json_path)
    json_files = [f for f in json_files if f.endswith(".json")]
    for json_file in json_files:
        data = json.load(open(os.path.join(ljspeech_json_path, json_file), "r"))["data"]
        for d in tqdm(data):
            name = d["wav"].split("/")[-1].replace(".wav", "")
            for n, p in zip(names, phones):
                if n == name:
                    d["phonemes"] = p
                    break

        json.dump({"data": data}, open(os.path.join(ljspeech_json_path, json_file), "w"), indent=1, ensure_ascii=False)

def find_min_max_values_in_sketch(metadata_root="data/dataset/metadata/ljspeech/phoneme_level"):
    print("Find min and max values in pitch and energy sketch.")
    pitch_dir = os.path.join(metadata_root, "pitch")
    energy_dir = os.path.join(metadata_root, "energy")
    stats_json = os.path.join(metadata_root, "stats.json")

    pitch_files = os.listdir(pitch_dir)
    energy_files = os.listdir(energy_dir)

    l_min = []
    l_max = []
    pitch_sketch_global_min = 1000
    pitch_sketch_global_max = -1000
    for f in tqdm(pitch_files):
        pitch = np.load(os.path.join(pitch_dir, f))
        pitch_sketch = sketch_extractor(pitch)
        p_min = pitch_sketch.min()
        p_max = pitch_sketch.max()
        if p_min < pitch_sketch_global_min:
            pitch_sketch_global_min = p_min
        if p_max > pitch_sketch_global_max:
            pitch_sketch_global_max = p_max
        l_min.append(p_min)
        l_max.append(p_max)
    print("pitch_sketch global min and max: ") 
    print(pitch_sketch_global_min, pitch_sketch_global_max)
    

    l_min = []
    l_max = []
    energy_sketch_global_min = 1000
    energy_sketch_global_max = -1000
    for f in tqdm(energy_files):
        energy = np.load(os.path.join(energy_dir, f))
        energy_sketch = sketch_extractor(energy)
        e_min = energy_sketch.min()
        e_max = energy_sketch.max()
        if e_min < energy_sketch_global_min:
            energy_sketch_global_min = e_min
        if e_max > energy_sketch_global_max:
            energy_sketch_global_max = e_max
        l_min.append(e_min)
        l_max.append(e_max)
    print("energy_sketch global min and max: ")
    print(energy_sketch_global_min, energy_sketch_global_max)

    with open(stats_json, "r") as f:
        stats = json.load(f)
        stats["pitch_sketch"] = [float(pitch_sketch_global_min), float(pitch_sketch_global_max)]
        stats["energy_sketch"] = [float(energy_sketch_global_min), float(energy_sketch_global_max)]
    
    with open(stats_json, "w") as f:
        f.write(json.dumps(stats))

if __name__ == "__main__":
    format_ljspeech("data/dataset/LJSpeech-1.1")

    cmd = "python drawspeech/utilities/preprocessor/preprocessor.py drawspeech/utilities/preprocessor/preprocess_phoneme_level.yaml"
    os.system(cmd)

    add_phoneme_for_ljspeech()
    
    find_min_max_values_in_sketch()

    