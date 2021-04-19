import librosa
import torchaudio
import numpy as np

# ------------ Preprocessing dataset audio ------------ #
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load("na_data/clips/" + batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch


def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 44_100, 16_000)
    batch["sampling_rate"] = 16_000
    return batch
