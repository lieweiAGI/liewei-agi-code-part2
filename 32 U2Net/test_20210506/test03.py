import torchaudio
import torch
import matplotlib.pyplot as plt

filename = r"waves_yesno\0_0_0_0_1_1_1_1.wav"

waveform,sample_rate = torchaudio.load(filename)
# print(waveform)
# print(sample_rate)
#
# plt.figure()
# plt.plot(waveform.t().numpy())
#时域转频域
# specgram = torchaudio.transforms.Spectrogram()(waveform)
# plt.figure()
# plt.imshow(specgram.log2()[0,:,:].numpy(),cmap="gray")

#梅尔频谱图
specgram = torchaudio.transforms.MelSpectrogram()(waveform)
plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(),cmap="gray")
plt.show()