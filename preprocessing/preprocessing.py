import os
import librosa
import soundfile as sf
import pandas as pd
import numpy as np


def convert_to_wav(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        wav_file_path = os.path.splitext(file_path)[0] + '.wav'
        sf.write(wav_file_path, y, sr, subtype='PCM_16')
        return wav_file_path
    except Exception as e:
        print(f"Errore durante la conversione in WAV del file '{file_path}': {e}")
        return None

def delete_non_wav_files(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            if not filename.endswith('.wav'):
                try:
                    os.remove(file_path)
                    print(f"File non-WAV eliminato: {file_path}")
                except Exception as e:
                    print(f"Errore durante l'eliminazione del file '{file_path}': {e}")


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

                    # Converti il file in WAV se non è già in WAV
                    if not file_path.endswith('.wav'):
                        file_path = convert_to_wav(file_path)
                        if not file_path:
                            continue  # Salta il file se la conversione fallisce

                    # Carica il file audio utilizzando librosa in mono
                    y, sr = librosa.load(file_path, sr=None, mono=True)

                    # Normalizza il segnale audio
                    y = librosa.util.normalize(y)

                    # Debugging print statement before conversion
                    print(f"Processing file: {filename}")
                    print(f"Loaded audio shape: {y.shape}, dtype: {y.dtype}")

                    # Converti y in formato 16-bit prima di scriverlo
                    y_16bit = np.int16(y * 32767)

                    # Sovrascrivi l'audio convertito come 16-bit con soundfile
                    sf.write(file_path, y_16bit, sr, subtype='PCM_16')

                    # Verifica i dettagli del file convertito
                    with sf.SoundFile(file_path) as f:
                        num_channels = f.channels
                        bit_depth = ' '.join(sf.info(file_path).subtype_info.split())

                    # Debugging print statements after conversion
                    print(f"Converted Channels: {num_channels}, Bit Depth: {bit_depth}")

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

                    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=512)
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
                        'Forma d\'onda': y.shape
                    })

                except Exception as e:
                    print(f"Errore durante l'elaborazione del file '{filename}': {e}")

    return audio_features, amplitudes, durations, frequencies, num_channels_list, phases, max_internal_frequencies, bit_depths

def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)



if __name__ == "__main__":
    dataset_root = "C:/Users/frees/Desktop/Dataset pulito senza duplicati"

    extracted_features, amplitudes, durations, frequencies, num_channels_list, phases, max_internal_frequencies, bit_depths = extract_audio_features(dataset_root)

    delete_non_wav_files(dataset_root)  # Elimina i file non WAV

    output_csv_file = '../datasets_csv/audio_features_dataset_prep.csv'

    save_to_csv(extracted_features, output_csv_file)

    print(f"Il file CSV '{output_csv_file}' è stato creato con successo.")