import torch
import torchaudio

from pathlib import Path
from third_party.audio import InverseSpectralNormalization, NormalizeDB, WaveRNNInferenceWrapper
from torchaudio.models import wavernn

from dapi_lib import models


text = "Hello world!"

# Initialize model, weights are optional
weights = models.Tacotron2Weights.Characters_WaveRNN_LJSpeech
model = models.tacotron2(weights=weights)
# model, weights = models.get('tacotron2', models.Tacotron2Weights.Characters_WaveRNN_LJSpeech)

model.eval()

# Initialize inference transforms
preprocess = weights.transforms()

# Apply inference preprocessing transforms
sequences, lengths = preprocess(text)

# Infer spectogram
with torch.no_grad():
    mel_specgram, _, _ = model.infer(sequences, lengths)

# Show number of symbols and spectrogram shape
print(weights.meta['n_symbol'], mel_specgram.shape)

# Use standard torchaudio code to convert spectrogram to wave file
wavernn_model = wavernn("wavernn_10k_epochs_8bits_ljspeech").eval()
wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model)
transforms = torch.nn.Sequential(InverseSpectralNormalization(), NormalizeDB(min_level_db=-100, normalization=True))
mel_specgram = transforms(mel_specgram)
with torch.no_grad():
    waveform = wavernn_inference_model(mel_specgram, mulaw=True, batched=True, timesteps=100, overlap=5)
Path("./output").mkdir(parents=True, exist_ok=True)
path = "./output/message.wav"
print(f"Saving wave at {path}")
torchaudio.save(path, waveform, sample_rate=22050)
