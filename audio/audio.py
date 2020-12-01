# Uncomment the following line to run in Google Colab
# !pip install torchaudio
import torch
import torchaudio
import requests
import matplotlib.pyplot as plt
url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
r = requests.get(url)
with open('data/test.wav', 'wb') as f:
    f.write(r.content)

filename = "data/test.wav"
waveform, sample_rate = torchaudio.load(filename)

waveform.shape
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())
plt.show()