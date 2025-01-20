import os
import librosa
import soundfile as sf
import pandas as pd


def extract_audio_features(root_folder):
    audio_features = []
    durations = []
    frequencies = []
    amplitudes = []
    num_channels_list = []
    phases = []
    max_internal_frequencies = []
    bit_depths = []

    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(root, filename)

                try:
                    file_path = str(file_path)

                    # Carica il file audio utilizzando librosa in mono
                    y, sr = librosa.load(file_path, sr=None, mono=True)

                    # Debugging print statement before extraction
                    print(f"Processing file: {file_path}")
                    print(f"Loaded audio shape: {y.shape}, dtype: {y.dtype}")

                    with sf.SoundFile(file_path) as f:
                        num_channels = f.channels
                        bit_depth = ' '.join(sf.info(file_path).subtype_info.split())

                    amplitude = max(abs(y))
                    duration = librosa.get_duration(y=y, sr=sr)
                    frequency = sr
                    phase = y[0]

                    amplitudes.append(amplitude)
                    durations.append(duration)
                    frequencies.append(frequency)
                    num_channels_list.append(num_channels)
                    phases.append(phase)
                    bit_depths.append(bit_depth)

                    # Calcolo dinamico di n_fft come una frazione della lunghezza del segnale
                    n_fft = min(2048, max(2, len(y) // 10))  # Imposta n_fft a un valore massimo di 2048 e almeno 2 campioni

                    # Estrazione delle caratteristiche usando il valore dinamico di n_fft
                    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=n_fft)
                    max_idx = magnitudes.argmax()
                    max_time_idx = max_idx // magnitudes.shape[0]
                    max_freq_idx = max_idx % magnitudes.shape[0]
                    max_internal_frequency = pitches[max_freq_idx, max_time_idx]
                    max_internal_frequencies.append(max_internal_frequency)

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
                        'n_fft': n_fft,
                        'Forma d\'onda': y.shape
                    })

                except Exception as e:
                    print(f"Errore durante l'elaborazione del file '{file_path}': {e}")

    return audio_features, amplitudes, durations, frequencies, num_channels_list, phases, max_internal_frequencies, bit_depths


def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def resample_dataset(dataset_path, sr):
    # Ottieni la lista dei file nel dataset
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)

                try:
                    # Carica il file audio
                    y, original_sr = librosa.load(file_path, sr=None)

                    # Ricampiona il file audio alla frequenza di campionamento desiderata
                    y_resampled = librosa.resample(y, orig_sr=original_sr, target_sr=sr)

                    # Sovrascrivi il file originale con quello ricampionato
                    sf.write(file_path, y_resampled, sr)

                    print(f"File {file_path} ricampionato a {sr} Hz e sovrascritto.")

                except Exception as e:
                    print(f"Errore durante il ricampionamento del file '{file_path}': {e}")


if __name__ == "__main__":
    dataset_root = "C:/Users/frees/Desktop/Dataset pulito senza duplicati Normalizzato_44100"

    # Resample the dataset before extracting features
    resample_dataset(dataset_root, 44100)

    # Extract features after resampling
    extracted_features, amplitudes, durations, frequencies, num_channels_list, phases, max_internal_frequencies, bit_depths = extract_audio_features(
        dataset_root)

    output_csv_file = '../datasets_csv/audio_features_dataset_44100.csv'
    save_to_csv(extracted_features, output_csv_file)
    print(f"Il file CSV '{output_csv_file}' è stato creato con successo.")
