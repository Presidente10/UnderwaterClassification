import os
import librosa
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import shutil

def extract_audio_features(root_folder):
    # Lista per memorizzare le informazioni estratte
    audio_features = []

    # Liste per memorizzare i valori delle caratteristiche
    durations = []
    frequencies = []
    amplitudes = []
    num_channels_list = []
    phases = []
    max_internal_frequencies = []
    bit_depths = []

    # Attraversa tutte le sottocartelle nel percorso radice
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(('.wav', '.mp3', '.flac')):  # Considera solo file audio WAV, MP3 o FLAC
                file_path = os.path.join(root, filename)

                try:
                    # Assicurati che file_path sia una stringa
                    file_path = str(file_path)

                    # Usa soundfile per ottenere informazioni sul file audio
                    with sf.SoundFile(file_path) as f:
                        num_channels = f.channels

                        bit_depth = ' '.join(sf.info(file_path).subtype_info.split())  # Estrai la profondità del bit

                    # Carica il file audio utilizzando librosa (senza convertire in mono)
                    y, sr = librosa.load(file_path, sr=None, mono=False)

                    # Se l'audio ha più di un canale, prendi solo il primo canale per le caratteristiche
                    if y.ndim > 1:
                        y = y[0]

                    # Estrai le informazioni richieste
                    amplitude = max(abs(y))  # Ampiezza massima assoluta del segnale
                    duration = librosa.get_duration(y=y, sr=sr)  # Durata del segnale in secondi
                    frequency = sr  # Frequenza di campionamento
                    phase = y[0]  # Fase del segnale (primo campione)

                    # Aggiungi i valori alle liste corrispondenti
                    amplitudes.append(amplitude)
                    durations.append(duration)
                    frequencies.append(frequency)
                    num_channels_list.append(num_channels)
                    phases.append(phase)
                    bit_depths.append(bit_depth)

                    # Estrazione della frequenza massima interna
                    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=512)
                    max_idx = magnitudes.argmax()
                    max_time_idx = max_idx // magnitudes.shape[0]
                    max_freq_idx = max_idx % magnitudes.shape[0]
                    max_internal_frequency = pitches[max_freq_idx, max_time_idx]
                    max_internal_frequencies.append(max_internal_frequency)

                    # Ottieni il nome del file senza l'estensione
                    file_name = os.path.splitext(filename)[0]
                    classe = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                    sottoclasse = os.path.basename(os.path.dirname(file_path))

                    audio_features.append({
                        'Nome file': file_name,
                        'Classe': classe,
                        'Sottoclasse': sottoclasse,
                        'Ampiezza del segnale': amplitude,
                        'Durata': duration,
                        'Frequenza': frequency,
                        'Numero di canali': num_channels,
                        'Fase': phase,
                        'Frequenza massima interna': max_internal_frequency,
                        'Bit Depth': bit_depth,
                        'Forma d\'onda': y.shape
                    })

                except Exception as e:
                    print(f"Errore durante l'elaborazione del file '{filename}': {e}")

    return audio_features, amplitudes, durations, frequencies, num_channels_list, phases, max_internal_frequencies, bit_depths


def create_clean_dataset(duplicates, root_folder, clean_folder):
    """
    Crea un nuovo dataset senza duplicati.

    Parameters:
        duplicates (DataFrame): DataFrame contenente i duplicati.
        root_folder (str): Percorso del dataset originale.
        clean_folder (str): Percorso del nuovo dataset senza duplicati.
    """
    os.makedirs(clean_folder, exist_ok=True)  # Crea la directory se non esiste
    duplicate_files = set(duplicates['Nome file'])
    copied_files = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            file_name = os.path.splitext(file)[0]  # Nome del file senza estensione
            if file_name not in duplicate_files:  # Copia solo file non duplicati
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, root_folder)  # Mantieni la struttura
                dest_path = os.path.join(clean_folder, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Crea le cartelle
                shutil.copy2(source_path, dest_path)  # Copia il file
                copied_files.append(dest_path)
                print(f"File copiato: {dest_path}")

    return copied_files


def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

def plot_audio_durations_histogram(durations, bins=100):
    plt.figure(figsize=(12, 8))
    sns.histplot(durations, bins=bins, kde=True)
    plt.title('Distribuzione delle durate degli audio')
    plt.xlabel('Durata (secondi)')
    plt.ylabel('Frequenza')
    plt.grid(True)
    plt.show()

def plot_audio_max_frequencies_histogram(max_internal_frequencies, bins=100):
    plt.figure(figsize=(12, 8))
    sns.histplot(max_internal_frequencies, bins=bins, kde=True)
    plt.title('Distribuzione delle frequenze massime ')
    plt.xlabel('Frequenza massima')
    plt.ylabel('Valore')
    plt.grid(True)
    plt.show()

def plot_distribution_boxplot(values, title):
    plt.figure(figsize=(10, 6))
    plt.plot(values)
    plt.title(title)
    plt.xlabel('Indice')
    plt.ylabel('Valore')
    plt.grid(True)
    plt.show()

def plot_distribution(values, title, x_label):
    # Conta le occorrenze di ciascun valore
    counter = Counter(values)

    # Ordina i valori unici
    unique_values = sorted(counter.keys())
    counts = [counter[value] for value in unique_values]

    # Crea un grafico di distribuzione utilizzando matplotlib
    plt.figure(figsize=(12, 8))
    bar_width = 0.8  # Larghezza delle barre
    indices = range(len(unique_values))

    bars = plt.bar(indices, counts, color='skyblue', edgecolor='black', width=bar_width)

    # Aggiunge le etichette alle barre
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), ha='center', va='bottom', fontsize=10)

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Occorrenze', fontsize=14)
    plt.xticks(indices, unique_values, rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_root = 'C:/Users/frees/Desktop/Dataset originale con duplicati'

    # Estrai le informazioni audio dal dataset
    extracted_features, amplitudes, durations, frequencies, num_channels_list, phases, max_internal_frequencies, bit_depths = extract_audio_features(
        dataset_root)

    # Specifica il percorso per salvare il file CSV di output
    output_csv_file = '../datasets_csv/audio_features_dataset.csv'

    # Salva le informazioni estratte in un file CSV
    save_to_csv(extracted_features, output_csv_file)
    print(f"Il file CSV '{output_csv_file}' è stato creato con successo.")
    clean_dataset_root = 'C:/Users/frees/Desktop/Dataset pulito senza duplicati'

    # Rimuovi i duplicati
    duplicates_df = pd.read_csv(output_csv_file)
    duplicates = duplicates_df[duplicates_df.duplicated(subset=['Nome file'], keep=False)]

    # Crea il nuovo dataset senza duplicati
    copied_files = create_clean_dataset(duplicates, dataset_root, clean_dataset_root)

    print(f"Creato un nuovo dataset senza duplicati: {clean_dataset_root}")
    print(f"Copiati {len(copied_files)} file unici.")
