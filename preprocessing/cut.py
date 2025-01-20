import os
import librosa
import soundfile as sf  # Importa soundfile per salvare file audio
import pandas as pd

def extract_audio_features(root_folder):
    audio_features = []
    segment_features = []

    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(root, filename)

                try:
                    file_path = str(file_path)

                    # Carica il file audio utilizzando librosa in mono
                    y, sr = librosa.load(file_path, sr=None, mono=True)


                    # Calcola la durata totale del file audio
                    duration = librosa.get_duration(y=y, sr=sr)

                    # Calcola il numero di segmenti da tagliare (ogni 4 secondi)
                    num_segments = int(duration // 4)
                    if duration % 4 != 0:
                        num_segments += 1

                    # Aggiungi le feature del file originale alla lista
                    audio_features.append({
                        'Nome file': os.path.splitext(filename)[0],
                        'Classe': os.path.basename(os.path.dirname(os.path.dirname(file_path))),
                        'Sottoclasse': os.path.basename(os.path.dirname(file_path)),
                        'Durata': duration,
                        'Numero di segmenti': num_segments,
                    })

                    # Taglia e aggiungi ogni segmento
                    for i in range(num_segments):
                        start_time = i * 4
                        end_time = min((i + 1) * 4, duration)  # Gestione dell'ultimo segmento

                        # Calcola gli indici di campione per il taglio
                        start_sample = int(start_time * sr)
                        end_sample = int(end_time * sr)

                        # Esegui il taglio
                        y_cut = y[start_sample:end_sample]

                        # Salva il segmento su disco utilizzando soundfile
                        segment_filename = f"{os.path.splitext(filename)[0]}_seg{i + 1}.wav"
                        segment_filepath = os.path.join(root, segment_filename)
                        sf.write(segment_filepath, y_cut, sr)

                        # Aggiungi le feature del segmento alla lista
                        segment_features.append({
                            'Nome file': segment_filename,
                            'Classe': os.path.basename(os.path.dirname(os.path.dirname(file_path))),
                            'Sottoclasse': os.path.basename(os.path.dirname(file_path)),
                            'Durata': librosa.get_duration(y=y_cut, sr=sr),
                            'Numero di segmento': i + 1,
                            'File di Provenienza': filename
                        })

                        # Debugging print statement after saving
                        print(f"Saved segment: {segment_filename}")

                except Exception as e:
                    print(f"Errore durante l'elaborazione del file '{file_path}': {e}")

    return audio_features, segment_features


def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    dataset_root = "C:/Users/frees/Desktop/Dataset pulito senza duplicati Normalizzato_44100_seg"

    # Estrai le features audio compresi i segmenti
    audio_features, segment_features = extract_audio_features(dataset_root)

    output_segment_features_file = '../datasets_csv/segment_features_dataset_44100.csv'

    save_to_csv(segment_features, output_segment_features_file)

    print(f"Il file CSV '{output_segment_features_file}' è stato creato con successo.")
