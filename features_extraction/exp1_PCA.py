import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # Importa tqdm per la barra di avanzamento


# Funzione per calcolare MFCC (ne calcola 13, poi selezioneremo gli indici desiderati)
def calculate_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)  # mfcc.shape = (13, time) => mean => (13,)


# Funzione per calcolare il Zero-Crossing Rate
def calculate_zcr(y):
    return np.mean(librosa.feature.zero_crossing_rate(y=y))


# Funzione per estrarre SOLO le MFCC desiderate (1,3,4,8,9,10,11,12,13) e ZCR
def extract_features(file_path):
    """
    Restituisce un array con:
      - MFCC 1,3,4,8,9,10,11,12,13  (in questo ordine, oppure in uno fisso)
      - ZCR
    """
    try:
        # Carichiamo l'audio
        y, sr = librosa.load(file_path, sr=None)

        # Calcolo di tutte le 13 MFCC
        mfccs = calculate_mfcc(y, sr)  # array di lunghezza 13
        zcr = calculate_zcr(y)

        # Indici corrispondenti alle MFCC che vogliamo tenere:
        # MFCC 1 => index 0
        # MFCC 3 => index 2
        # MFCC 4 => index 3
        # MFCC 8 => index 7
        # MFCC 9 => index 8
        # MFCC 10 => index 9
        # MFCC 11 => index 10
        # MFCC 12 => index 11
        # MFCC 13 => index 12
        selected_indices = [0, 2, 3, 7, 8, 9, 10, 11, 12]
        mfcc_selected = mfccs[selected_indices]

        # Combiniamo le feature scelte in un unico array
        # Prima le MFCC scelte, poi ZCR
        features_array = np.concatenate([mfcc_selected, [zcr]])
        return features_array
    except Exception as e:
        print(f"Errore durante l'elaborazione di {file_path}: {str(e)}")
        return None


def extract_features_from_directory(audio_directory):
    """
    Estrae SOLO:
      - MFCC 1,3,4,8,9,10,11,12,13
      - ZCR
    da tutti i file .wav presenti in audio_directory (e sottocartelle).
    """
    features_list = []
    file_names = []
    classes = []
    subclasses = []

    # Raccogliamo i file
    files = []
    for root, _, filenames in os.walk(audio_directory):
        for filename in filenames:
            if filename.endswith('.wav'):
                files.append(os.path.join(root, filename))

    total_files = len(files)
    processed_files = 0
    percent_complete = 0

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

            # Ottieni classe e sottoclasse dalla struttura delle cartelle
            rel_path = os.path.relpath(file_path, audio_directory)
            path_parts = rel_path.split(os.sep)
            class_name = path_parts[0] if len(path_parts) > 1 else 'Unknown'
            subclass_name = path_parts[1] if len(path_parts) > 2 else 'Unknown'
            classes.append(class_name)
            subclasses.append(subclass_name)

    # Definiamo le colonne
    # MFCC 1,3,4,8,9,10,11,12,13 + ZCR
    # Usando un ordine coerente: (1,3,4,8,9,10,11,12,13, ZCR)
    # Bada che "MFCC 1" in Python => index 0 => stiamo mappando. L'ordine è per coerenza
    columns = [
        'MFCC 1', 'MFCC 3', 'MFCC 4', 'MFCC 8',
        'MFCC 9', 'MFCC 10', 'MFCC 11', 'MFCC 12',
        'MFCC 13', 'ZCR'
    ]

    # Creiamo il DataFrame
    df = pd.DataFrame(features_list, columns=columns)
    df.insert(0, 'File Name', file_names)
    df.insert(1, 'Class', classes)
    df.insert(2, 'Subclass', subclasses)

    return df


def main():
    audio_directory = 'C:/Users/frees/Desktop/Dataset pulito senza duplicati Normalizzato_44100_seg'
    df = extract_features_from_directory(audio_directory)

    if not df.empty:
        df.to_csv('exp1_audio_features_44100_PCA.csv', index=False)
        print(f"Caratteristiche estratte da {len(df)} file audio e salvate in 'exp1_audio_features.csv'.")
    else:
        print("Nessuna caratteristica estratta.")


if __name__ == "__main__":
    main()
