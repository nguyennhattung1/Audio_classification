from preprocessing_data import AudioUtil
import torch
filepath = "7061-6-0-0.wav"



aud = AudioUtil.open(filepath)
reaud = AudioUtil.resample(aud, 44100)
rechan = AudioUtil.rechannel(reaud, 1)

dur_aud = AudioUtil.pad_trunc(rechan, 4000)
shift_aud = AudioUtil.time_shift(dur_aud, 0.4)
sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
#Convert the spectrogram to a PyTorch tensor and add batch dimension
input_tensor = torch.tensor(aug_sgram, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
print(input_tensor.shape)