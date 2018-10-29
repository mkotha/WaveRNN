import sys
import glob
import pickle
import os
import multiprocessing as mp
from utils.dsp import *

SEG_PATH = sys.argv[1]
DATA_PATH = sys.argv[2]

def get_files(path):
    next_speaker_id = 0
    speaker_ids = {}
    filenames = []
    for filename in glob.iglob(f'{path}/**/*.wav', recursive=True):
        speaker_name = filename.split('/')[-2]
        if speaker_name not in speaker_ids:
            speaker_ids[speaker_name] = next_speaker_id
            next_speaker_id += 1
            filenames.append([])
        filenames[speaker_ids[speaker_name]].append(filename)

    return filenames

files = get_files(SEG_PATH)

def process_file(i, path):
    dir = f'{DATA_PATH}/{i}'
    name = path.split('/')[-1][:-4] # Drop .wav
    filename = f'{dir}/{name}.npy'
    if os.path.exists(filename):
        print(f'{filename} already exists, skipping')
        return
    floats = load_wav(path, encode=False)
    trimmed, _ = librosa.effects.trim(floats, top_db=25)
    quant = (trimmed * (2**15 - 0.5) - 0.5).astype(np.int16)
    if max(abs(quant)) < 2048:
        print(f'audio fragment too quiet ({max(abs(quant))}), skipping: {path}')
        return
    if len(quant) < 10000:
        print(f'audio fragment too short ({len(quant)} samples), skipping: {path}')
        return
    os.makedirs(dir, exist_ok=True)
    np.save(filename, quant)
    return name

index = []
with mp.Pool(8) as pool:
    for i, speaker in enumerate(files):
        res = pool.starmap_async(process_file, [(i, path) for path in speaker]).get()
        index.append([x for x in res if x])
        print(f'Done processing speaker {i}')

os.makedirs(DATA_PATH, exist_ok=True)
with open(f'{DATA_PATH}/index.pkl', 'wb') as f:
    pickle.dump(index, f)
