import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # Importa tqdm per la barra di avanzamento


# Funzione per calcolare MFCC
def calculate_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)


# Funzione per calcolare il Zero-Crossing Rate
def calculate_zcr(y):
    return np.mean(librosa.feature.zero_crossing_rate(y=y))


# Funzione per calcolare il Spectral Centroid
def calculate_spectral_centroid(y, sr):
    return np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))


# Funzione per calcolare la Spectral Bandwidth
def calculate_spectral_bandwidth(y, sr):
    return np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))


# Funzione per calcolare Chroma Features
def calculate_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)


# Funzione per estrarre tutte le caratteristiche
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        mfccs = calculate_mfcc(y, sr)
        zcr = calculate_zcr(y)
        spectral_centroid = calculate_spectral_centroid(y, sr)
        spectral_bandwidth = calculate_spectral_bandwidth(y, sr)
        chroma = calculate_chroma(y, sr)

        return [*mfccs, zcr, spectral_centroid, spectral_bandwidth, *chroma]
    except Exception as e:
        print(f"Errore durante l'elaborazione di {file_path}: {str(e)}")
        return None


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

    total_files = len(files)
    processed_files = 0
    percent_complete = 0

    # Usa tqdm per mostrare la barra di avanzamento
    for file_path in tqdm(files, desc="Elaborazione file audio", unit="file", position=0, leave=True):
        processed_files += 1
        new_percent_complete = int((processed_files / total_files) * 100)

        if new_percent_complete > percent_complete:
            percent_complete = new_percent_complete
            print(f"Progress: {percent_complete}%")

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

    columns = [f'MFCC {i + 1}' for i in range(13)] + ['ZCR', 'Spectral Centroid', 'Spectral Bandwidth'] + [
        f'Chroma {i + 1}' for i in range(12)]

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
        df.to_csv('exp1_audio_features.csv', index=False)
        print(f"Caratteristiche estratte da {len(df)} file audio e salvate in 'exp1_audio_features.csv'.")
    else:
        print("Nessuna caratteristica estratta.")


if __name__ == "__main__":
    main()
