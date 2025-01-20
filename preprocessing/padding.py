import os
from pydub import AudioSegment
import pandas as pd

# Directory principale contenente le sottocartelle
#audio_directory = "C://Users//mario//OneDrive//Desktop//Dataset_segmentato//"  # Sostituisci con il percorso della tua directory
# Nome del file CSV di output
#output_csv = "audio_features_padding.csv"
audio_directory = "C:/Users/frees/Desktop/Dataset pulito senza duplicati Normalizzato_44100_seg"
output_csv = "segment_features_dataset_44100_pad.csv"


# Funzione per aggiungere silenzio ai file audio inferiori a 4 secondi
def add_silence(audio_file):
    audio = AudioSegment.from_file(audio_file)
    original_duration = len(audio)

    if original_duration < 4000:
        silence_duration = 4000 - original_duration
        silence = AudioSegment.silent(duration=silence_duration)
        padded_audio = audio + silence
        padded_audio.export(audio_file, format="wav")  # Sovrascrivi il file originale
        print(f"Aggiunto silenzio a {audio_file}. Nuova durata: {len(padded_audio)} ms")
    elif original_duration > 4000:
        print(f"Il file {audio_file} è già più lungo di 4 secondi. Nessuna modifica necessaria.")
    else:
        print(f"Il file {audio_file} è esattamente di 4 secondi. Nessuna modifica necessaria.")

# Funzione per elaborare i file audio e raccogliere i dati
def process_audio_files_and_create_csv(directory, output_file):
    data = []
    file_count = 0  # Contatore per contare i file aggiunti al CSV

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".wav"):
                if "_seg" not in filename:
                    os.remove(os.path.join(root, filename))  # Rimuovi i file che non contengono "_seg" nel nome
                    continue

                file_path = os.path.join(root, filename)
                audio = AudioSegment.from_file(file_path)
                duration = len(audio)

                if duration > 4000:
                    os.remove(file_path)  # Elimina i file con durata maggiore di 4 secondi
                elif duration < 4000:
                    add_silence(file_path)  # Aggiungi silenzio e sovrascrivi
                    audio = AudioSegment.from_file(file_path)
                    duration = len(audio)  # Ricarica l'audio per aggiornare la durata

                # Aggiungi i dati al dataframe solo se il file è valido (durata <= 4000)
                if duration <= 4000:
                    file_name = os.path.splitext(filename)[0]
                    classe = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                    sottoclasse = os.path.basename(os.path.dirname(file_path))

                    data.append({
                        "Nome file": file_name,
                        "Classe": classe,
                        "Sottoclasse": sottoclasse,
                        "Durata": duration,
                    })

                    file_count += 1  # Incrementa il contatore di file aggiunti al CSV
                    print(f"File aggiunto al CSV: {file_name}. Numero totale di file aggiunti: {file_count}")

    # Crea il file CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"Processo completato. Numero totale di file aggiunti al CSV: {file_count}")

# Funzione principale
def main():
    # Elabora i file audio e crea il CSV
    process_audio_files_and_create_csv(audio_directory, output_csv)

    print(f"Il file CSV è stato salvato come {output_csv} e i file audio sono stati modificati nel dataset.")

if __name__ == "__main__":
    main()
