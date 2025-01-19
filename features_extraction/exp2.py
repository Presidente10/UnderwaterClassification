import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from librosa.feature import tempo


# Funzione per calcolare MFCC
def calculate_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# Funzione per calcolare il Zero-Crossing Rate
def calculate_zcr(y):
    return np.mean(librosa.feature.zero_crossing_rate(y=y))

# Funzione per calcolare Spectral Contrast
def calculate_spectral_contrast(y, sr):
    return np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

# Funzione per calcolare Tonnetz
def calculate_tonnetz(y, sr):
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    return np.mean(tonnetz, axis=1)

# Funzione per calcolare Chroma Features
def calculate_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)

# Funzione per calcolare il tempo
def calculate_tempo(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_value = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
    return np.mean(tempo_value)

# Funzione per estrarre tutte le caratteristiche
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        mfccs = calculate_mfcc(y, sr)
        zcr = calculate_zcr(y)
        spectral_contrast = calculate_spectral_contrast(y, sr)
        tonnetz = calculate_tonnetz(y, sr)
        chroma = calculate_chroma(y, sr)
        tempo = calculate_tempo(y, sr)

        return [*mfccs, zcr, spectral_contrast, *tonnetz, *chroma, tempo]
    except Exception as e:
        print(f"Errore durante l'elaborazione di {file_path}: {str(e)}")
        return None

# Funzione per estrarre caratteristiche da una directory
def extract_features_from_directory(audio_directory):
    features_list = []
    file_names = []
    classes = []
    subclasses = []

    # Ottieni una lista di tutti i file audio nella directory
    files = []
    for root, _, filenames in os.walk(audio_directory):
        for filename in filenames:
            if filename.endswith('.wav'):
                files.append(os.path.join(root, filename))

    # Usa tqdm per mostrare la barra di avanzamento
    for file_path in tqdm(files, desc="Elaborazione file audio", unit="file"):
        #print(f"Processing file: {os.path.basename(file_path)}")
        features = extract_features(file_path)
        if features is not None:
            features_list.append(features)
            file_names.append(os.path.basename(file_path))
            # Ottieni classe e sottoclasse dalla struttura della directory
            rel_path = os.path.relpath(file_path, audio_directory)
            path_parts = rel_path.split(os.sep)
            class_name = path_parts[0] if len(path_parts) > 1 else 'Unknown'
            subclass_name = path_parts[1] if len(path_parts) > 2 else 'Unknown'
            classes.append(class_name)
            subclasses.append(subclass_name)

    columns = [f'MFCC {i + 1}' for i in range(13)] + ['ZCR', 'Spectral Contrast'] + [f'Tonnetz {i + 1}' for i in range(6)] + [f'Chroma {i + 1}' for i in range(12)] + ['Tempo']

    # Aggiungi colonne per Class e Subclass
    columns = ['File Name', 'Class', 'Subclass'] + columns

    df = pd.DataFrame(features_list, columns=columns[3:])
    df.insert(0, 'File Name', file_names)
    df.insert(1, 'Class', classes)
    df.insert(2, 'Subclass', subclasses)

    return df

# Funzione principale
def main():
    audio_directory = 'C:/Users/frees/Desktop/DATASET_SEGMENTATO_88'
    df = extract_features_from_directory(audio_directory)

    if not df.empty:
        df.to_csv('exp2_audio_features.csv', index=False)
        print(f"Caratteristiche estratte da {len(df)} file audio e salvate in 'exp2_audio_features.csv'.")
    else:
        print("Nessuna caratteristica estratta.")

if __name__ == "__main__":
    main()
